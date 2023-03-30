import pytest
from functools import partial

import numpy as np
import random

import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.tensor import ColoParameter, ProcessGroup, ShardSpec, ComputePattern, ComputeSpec, \
    ColoTensor, ColoTensorSpec
from colossalai.nn.parallel.layers import CachedParamMgr, CachedEmbeddingBag, ParallelCachedEmbeddingBag, EvictionStrategy, \
    ParallelCachedEmbeddingBagTablewise, TablewiseEmbeddingBagConfig
from typing import List

NUM_EMBED, EMBED_DIM = 10, 8
BATCH_SIZE = 8


def set_seed(seed):
    """
    To achieve reproducible results, it's necessary to fix random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthesize_1d_sparse_feature(
    batch_size,
    num_embed,
    device,
):
    indices_in_batch = batch_size * 2
    indices = torch.randint(low=0, high=num_embed, size=(indices_in_batch,), device=device, dtype=torch.long)
    offsets = torch.from_numpy(
        np.array([
            0, *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))), indices_in_batch
        ])).to(device).long()
    return indices, offsets


@pytest.mark.skip
def test_cachemgr():
    model = torch.nn.EmbeddingBag(10000, 128)
    # 10 chunks, 5 in cuda
    mgr = CachedParamMgr(model.weight.detach(), 5)
    assert mgr.cuda_row_num == 5

    mgr._admit(1)
    assert not mgr._chunk_in_cuda(2)
    assert mgr._chunk_in_cuda(1)

    # print(mgr.cached_chunk_table)
    mgr._admit(8)

    # now 3 chunk is available
    assert mgr.cuda_available_chunk_num == 3

    mgr._evict()
    assert mgr.cuda_available_chunk_num == 4

    mgr._prepare_rows_on_cuda(torch.tensor([9, 6, 5], dtype=torch.long, device=0))
    mgr._prepare_rows_on_cuda(torch.tensor([3, 4, 5], dtype=torch.long, device=0))
    # print(mgr.cached_chunk_table)
    # mgr.print_comm_stats()

    mgr.flush()
    assert mgr.cuda_available_chunk_num == 5


def test_reorder_with_freq():
    num_embed = 100
    chunk_size = 1
    num_chunk = 5

    idx_map = torch.randint(10000, size=(num_embed,))
    sorted_idx = torch.argsort(idx_map, descending=True).tolist()
    chunkid, offset_in_chunk = [], []
    for i in range(num_embed):
        idx = sorted_idx.index(i)
        chunkid.append(idx // chunk_size)
        offset_in_chunk.append(idx % chunk_size)

    dev = torch.device('cuda')
    chunkid = torch.tensor(chunkid, dtype=torch.long, device=dev)
    offset_in_chunk = torch.tensor(offset_in_chunk, dtype=torch.long, device=dev)

    weight = torch.rand(num_embed, 2)
    mgr = CachedParamMgr(weight, num_chunk)

    mgr.reorder(idx_map)

    indices = mgr.idx_map.index_select(0, torch.arange(num_embed, dtype=torch.long, device=dev))
    mgr_chunk_id = torch.div(indices, chunk_size, rounding_mode='floor')
    mgr_offsets = torch.remainder(indices, chunk_size)
    assert torch.allclose(chunkid, mgr_chunk_id), f"chunk id: {chunkid}, mgr: {mgr_chunk_id}"
    assert torch.allclose(offset_in_chunk, mgr_offsets), \
        f"offset in chunk: {offset_in_chunk}, mgr: {mgr_offsets}"


@pytest.mark.parametrize('use_LFU', [True, False])
def test_freq_aware_embed(use_LFU: bool):
    device = torch.device('cuda', 0)
    evict_strategy = EvictionStrategy.LFU if use_LFU else EvictionStrategy.DATASET
    model = CachedEmbeddingBag(NUM_EMBED,
                               EMBED_DIM,
                               mode='mean',
                               include_last_offset=True,
                               cache_ratio=min(BATCH_SIZE * 2 / NUM_EMBED, 1.0),
                               ids_freq_mapping=None,
                               evict_strategy=evict_strategy).to(device)

    assert model.weight.shape[0] == NUM_EMBED
    ref_model = torch.nn.EmbeddingBag.from_pretrained(model.weight.detach().to(device),
                                                      mode='mean',
                                                      include_last_offset=True,
                                                      freeze=False)

    assert torch.allclose(ref_model.weight.detach(), model.weight.detach().to(device))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for i in range(5):
        indices, offsets = synthesize_1d_sparse_feature(BATCH_SIZE, NUM_EMBED, device)
        res = model(indices, offsets)
        ref_res = ref_model(indices, offsets)
        assert torch.allclose(res, ref_res), f"model result: {res}, reference: {ref_res}"

        grad = torch.rand_like(res)
        # comparing gradient here is nontrivial
        res.backward(grad)
        ref_res.backward(grad)
        optimizer.step()
        optimizer.zero_grad()

        ref_optimizer.step()
        ref_optimizer.zero_grad()

    model.cache_weight_mgr.flush()
    model_weight = model.weight.detach().to(device)
    ref_weight = ref_model.weight.detach()
    assert torch.allclose(model_weight, ref_weight), \
        f"model weight: {model_weight[10:18, :8]}, reference: {ref_weight[10:18, :8]}"


@pytest.mark.parametrize('init_freq', [True, False])
def test_lfu_strategy(init_freq: bool):
    # minimal test to check behavior
    Bag = CachedEmbeddingBag(5,
                             5,
                             cache_ratio=3 / 5,
                             buffer_size=0,
                             pin_weight=True,
                             ids_freq_mapping=[4, 2, 1, 3, 1] if init_freq else None,
                             warmup_ratio=1.0,
                             evict_strategy=EvictionStrategy.LFU)

    # print('cached_idx_map: ', Bag.cache_weight_mgr.cached_idx_map)
    offsets = torch.tensor([0], device="cuda:0")

    # prepare frequency learning info:
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)

    # check strategy
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([3], device="cuda:0"), offsets)    # miss, evict 1
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)    # hit
    Bag.forward(torch.tensor([4], device="cuda:0"), offsets)    # miss, evict 3
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)    # hit
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)    # hit

    assert torch.allclose(torch.Tensor(Bag.cache_weight_mgr.num_hits_history[-6:]), torch.Tensor([3, 0, 1, 0, 1, 1])), \
        "LFU strategy behavior failed"


def gather_tensor(tensor, rank, world_size):
    gather_list = []
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    torch.distributed.gather(tensor, gather_list, dst=0)
    return gather_list


def run_parallel_freq_aware_embed_tablewise(rank, world_size):
    if world_size != 2:
        return
    device = torch.device('cuda', torch.cuda.current_device())

    # initialize weight
    # 3 feature tables. idx: 0~5, 6~10, 11~17
    weight_tables = torch.rand(18, 5)
    weight_table1 = weight_tables[0:6]
    weight_table2 = weight_tables[6:11]
    weight_table3 = weight_tables[11:18]
    embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(num_embeddings=6,
                                    cuda_row_num=4,
                                    assigned_rank=0,
                                    initial_weight=weight_table1.clone().detach().cpu()))
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(num_embeddings=5,
                                    cuda_row_num=4,
                                    assigned_rank=0,
                                    initial_weight=weight_table2.clone().detach().cpu()))
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(num_embeddings=7,
                                    cuda_row_num=4,
                                    assigned_rank=1,
                                    initial_weight=weight_table3.clone().detach().cpu()))
    if rank == 0:
        _weight = torch.cat([weight_table1, weight_table2], 0)
    else:
        _weight = weight_table3
    model = ParallelCachedEmbeddingBagTablewise(
        embedding_bag_config_list,
        embedding_dim=5,
        _weight=_weight,
        include_last_offset=True,
        cache_ratio=0.5,
        buffer_size=0,
        evict_strategy=EvictionStrategy.LFU,
    )
    # explain
    '''
    batch       feature 1       feature 2       feature 3
    input0      [1,2,3]         [6,7]           []
    input1      []              [9]             [13,15]
    input2      [1,5]           [6,8]           [11]
                  ↑               ↑               ↑ 
                rank 0          rank 0          rank 1
    in KJT format
    '''
    res = model(torch.tensor([1, 2, 3, 1, 5, 6, 7, 9, 6, 8, 13, 15, 11], device=device),
                torch.tensor([0, 3, 3, 5, 7, 8, 10, 10, 12, 13], device=device),
                already_split_along_rank=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    rand_grad = torch.rand(3, 5 * 3, dtype=res.dtype, device=res.device)
    if rank == 0:
        fake_grad = rand_grad[0:2]
    else:
        fake_grad = rand_grad[2:]
    res.backward(fake_grad)
    optimizer.step()
    optimizer.zero_grad()

    # check correctness
    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(weight_tables.detach().clone(),
                                                          include_last_offset=True,
                                                          freeze=False).to(device)
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
        ref_fake_grad = torch.cat(rand_grad.split(5, 1), 0)
        ref_res = ref_model(torch.tensor([1, 2, 3, 1, 5, 6, 7, 9, 6, 8, 13, 15, 11], device=device),
                            torch.tensor([0, 3, 3, 5, 7, 8, 10, 10, 12, 13], device=device))
        ref_res.backward(ref_fake_grad)
        ref_optimizer.step()
        ref_optimizer.zero_grad()

        model.cache_weight_mgr.flush()
        recover_weight = model.cache_weight_mgr.weight.to(device)
        ref_weight = ref_model.weight.detach()[:11]
        assert torch.allclose(recover_weight, ref_weight), f"{recover_weight - ref_weight}"


def run_parallel_freq_aware_embed_columnwise(rank, world_size):
    device = torch.device('cuda', torch.cuda.current_device())

    num_embed = 100
    embed_dim = 16
    batch_size = 4

    set_seed(4321)
    weight = torch.rand(num_embed, embed_dim)
    coloweight = ColoTensor(weight.clone().detach().cpu(), spec=None)

    # initialize the tensor spec for the embedding weight parameter,
    # which is an ColoParameter.
    coloweight.set_process_group(ProcessGroup(tp_degree=world_size))
    coloweight.set_tensor_spec(ShardSpec(dims=[-1], num_partitions=[world_size]), ComputeSpec(ComputePattern.TP1D))

    model = ParallelCachedEmbeddingBag.from_pretrained(
        coloweight,
        include_last_offset=True,
        freeze=False,
        cache_ratio=batch_size * 2 / num_embed,
    )

    assert model.cache_weight_mgr.weight.device.type == 'cpu'
    assert model.cache_weight_mgr.cuda_cached_weight.requires_grad
    weight_in_rank = torch.tensor_split(weight, world_size, -1)[rank]
    print(f"model weight: {model.cache_weight_mgr.weight.shape}, ref weight: {weight_in_rank.shape}")
    assert torch.allclose(weight_in_rank,
                          model.cache_weight_mgr.weight.detach()), f"{weight_in_rank - model.cache_weight_mgr.weight}"

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(weight.detach().clone(),
                                                          include_last_offset=True,
                                                          freeze=False).to(device)
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    set_seed(4321)
    for i in range(5):
        indices, offsets = synthesize_1d_sparse_feature(batch_size, num_embed, device)
        res = model(indices, offsets)

        grad = torch.rand(batch_size * 2, embed_dim, dtype=res.dtype, device=res.device)
        grad_in_rank = torch.tensor_split(grad, world_size, 0)[rank]
        res.backward(grad_in_rank)

        optimizer.step()
        optimizer.zero_grad()

        res_list = gather_tensor(res.detach(), rank, world_size)

        if rank == 0:
            ref_res = ref_model(indices, offsets)
            recover_res = torch.cat(res_list, dim=0)

            assert torch.allclose(ref_res, recover_res)

            ref_res.backward(grad)
            ref_optimizer.step()
            ref_optimizer.zero_grad()

    model.cache_weight_mgr.flush()
    weight_list = gather_tensor(model.cache_weight_mgr.weight.detach().cuda(), rank, world_size)
    if rank == 0:
        recover_weight = torch.cat(weight_list, dim=1)
        assert torch.allclose(recover_weight, ref_model.weight.detach()), f"{recover_weight - ref_model.weight}"


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    # run_parallel_freq_aware_embed_columnwise(rank, world_size)
    run_parallel_freq_aware_embed_tablewise(rank, world_size)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_parallel_freq_aware_embed(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

def test_cached_embedding_bag_reloading():
    '''
    Test scenario: Find out if the model can continue the trend of the loss
        Step 1: Train the model for a number steps
        Step 2. Save the model as checkpoint
        Step 3. Load the model from checkpoint
        Step 4. Continue training
    '''
    from pathlib import Path
    import tqdm

    class RandomNegativeSampler():
        def __init__(self, item_probs, pool_size=1000000):
            self.random = np.random.default_rng(11111)
            self.item_probs = np.array(item_probs)/sum(item_probs)
            self.pool_size = pool_size
            self.all_items = np.arrage(len(item_probs))
            self.pool = self.random.choice(self.all_items, size=self.pool_size, 
                            p=self.item_probs, replace=True)
        
        def get_sample(self,sample_size):
            return self.random.choice(self.pool, size=sample_size, relace=True)

    class RecModel(torch.nn.Module):
        def __init__(self, total_no_items, emb_size, interm_size, output_size, layers, emb_weight=None):
            super().__init__()

            self.negatives = 20

            item_probs = [0, 0] + np.random.rand(total_no_items-2)

            self.sampler = RandomNegativeSampler(item_probs, total_no_items)

            self.embedding_layer = CachedEmbeddingBag(total_no_items, emb_size, _weight=emb_weight,
                                        dtype=torch.bfloat16, cache_ratio=0.02)
            self.mappings = torch.nn.ModuleList([torch.nn.Linear(emb_size, interm_size) 
                                        for _ in range(layers)])

            self.projection = torch.nn.Linear(emb_size, output_size)

            self.activation = torch.nn.GELU()

            self.loss = torch.nn.CrossEntropyLoss()
        
        def mask(self, inputs, prob=0.2):
            '''Randomly mask some items'''

            labels = inputs.clone()
            rnd = torch.rand(size=labels.shape, device=labels.device)

            input_mask = (rnd < prob) & (labels != 0)
            labels = labels * input_mask
            masked_inputs = 1 * input_mask + inputs * (1-1*input_mask)
            negatives = self.sampler.get_sample(labels.shape[0] * labels.shape[1] * self.negatives)
            negatives = negatives.reshape(labels.shape[0], labels.shape[1], self.negatives)
            labels = torch.cat([labels.unsqueeze(2), negatives], dim=2)

            return masked_inputs, labels

        def forward(self, X):
            '''X is 2D tensor: batch size x number_items'''

            X, labels = self.mask(X)

            x_2d_shape = X.shape
            X = X.view(1, -1).squeeze(dim=0) # To 1D
            offsets = torch.arange(X.shape[0], device=X.device)
            emb_1d_shape = torch.cat([torch.tensor(x_2d_shape), torch.tensor([-1])])

            embeddings = self.embedding_layer(X, offsets).to(torch.float32).view(torch.size(emb_1d_shape))

            for mapping in self.mappings:
                embeddings = self.activation(mapping(embeddings))
            
            embeddings = self.projection(embeddings)

            x_2d_shape = labels.shape
            X = labels.view(1, -1).squeeze(dim=0) # To 1D
            offsets = torch.arange(labels.shape[0], device=labels.device)
            emb_1d_shape = torch.cat([torch.tensor(x_2d_shape), torch.tensor([-1])])

            label_embeddings = self.embedding_layer(labels, offsets).to(torch.float32).view(torch.size(emb_1d_shape))

            result = torch.einsum("bse, bsne -> bsn", embeddings, label_embeddings)

            ground_truth = torch.zeros(size=result.shape, device=result.device)
            ground_truth[:,:, 0] = 1
            loss = self.loss(result.swapaxes(1,2), target=groud_truth.swapaxes(1,2))

            is_masked = (1* (X == 1)).to(X.device)
            loss = loss * is_masked / is_masked.sum()

            return loss

    class DataGenerator:
        def __init__(self, total_no_items, items_per_batch, batches):
            self.total_no_items = total_no_items
            self.items_per_batch = items_per_batch # Should be multiples of 6
            self.repeats = self.items_per_batch // 6
            self.batches = batches

        def _generate_record(self):
            '''Generate the record on the fly'''
            record = np.zeros(self.items_per_batch)
            for i in range(self.repeats):
                # Generate 5 random numbers
                start_numbers = np.random.randint(low=2, high=total_no_items, size=5)
                # Then pick one from one of them with probability
                prob = np.array([0.05, 0.1, 0.15, 0.2, 0.5])
                #Perturbate the probability with a noise and normalize them
                prob += np.random.rand(1) * 0.05
                prob /= prob.sum()

                pick_number = np.random.choice(start_numbers, p=prob)
                record[i*6:i*6+5] = start_numbers.copy()
                record[i*6+5] = pick_number.copy()
            
            return record
        
        def next(self):
            batches = []
            for i in range(self.batches):
                batches.append(self._generate_record())
            
            return torch.tensor(np.stack(batches))


    def train(checkpoint_file="", max_batches=100000):
        '''Training the model for a specific number of batches'''

        # Check if there is an embedding weight stored in checkpoint file
        cpu_weight = None
        checkpoint_dict = None

        if checkpoint_file is not None and len(checkpoint_file)>0:
            if Path(emb_checkpoint_file).exists():
                checkpoint_dict = torch.load(emb_checkpoint_file, map_location='cpu')

                cpu_weight = checkpoint_dict['cpu_weight']
        
        total_no_items, emb_size, interm_size, output_size, layers = 1000000, 128, 512, 128, 5

        # Initiate the model, which should be on GPU because of CachedEmbeddingBag
        model = RecModel(total_no_items, emb_size, interm_size, output_size, layers,
                        emb_weight=cpu_weight).to('cuda:0')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, wd=0.001)
        scheduler = torch.optim_lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_batches)
        
        # Now reload the rest model parameters if available
        if checkpoint_dict is not None:
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

        batch_count = 0
        batches_for_saving = 100
        pbar = tqdm.tqdm(total=batches_for_saving, ascii=True)

        items_per_batch, batches = 120, 32
        train_generator = DataGenerator(total_no_items, items_per_batch, batches)

        while True:
            model.train()
            optimizer.zero_grad()

            batch = train_generator.next()

            loss = model(batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.update(1)
            pbar.set_postfix(loss=np.mean(loss))
            batch_count += 1

            if batch_count > 0 and batch_count % batches_for_saving == 0:
                model_dict = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict(),
                              "cpu_weight": model.embedding_layer.cache_weight_mgr.weight
                            }
                
                torch.save(model_dict, checkpoint_file)


            if batch_count >= max_batches:
                # Last save and exit
                model_dict = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict(),
                              "cpu_weight": model.embedding_layer.cache_weight_mgr.weight
                            }
                
                torch.save(model_dict, checkpoint_file)

                break

    # Now chain all those together to similate my application scenarios

    # Step 1: Train the model for 10000 steps
    # Please pay attention to the loss by the end of the training
    checkpoint_file = "./my_model_data.chkp"
    train(checkpoint_file=checkpoint_file, max_batches=10000)

    # Step 2: Now reload the trained model from the checkpoint file and resume
    # training for 100000
    # Please pay attention to the loss at the beginning of the training
    # It should be similar to the loss of Step 1 by the endo the training
    # However, that is not the case: the loss is much higher than that.
    # My best guess is the cpu_weight is not probably propagated to cuda_cached_weight
    # or the cpu_weight does not capture the trained by the end of Step 1
    train(checkpoint_file=checkpoint_file, max_batches=10000)



if __name__ == '__main__':
    # test_freq_aware_embed(True)
    test_parallel_freq_aware_embed(2)
    # test_lfu_strategy(False)
