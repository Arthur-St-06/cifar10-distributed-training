from dataloader import get_dataloader

def test_dataloader_batch():
    dl = get_dataloader(batch_size=16, rank=0, world_size=1)
    x, y = next(iter(dl))
    assert x.shape[0] == 16, "Batch size should be 16"
    assert x.shape[1:] == (3, 32, 32), "Input shape should be (3, 32, 32)"