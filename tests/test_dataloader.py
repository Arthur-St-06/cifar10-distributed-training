from src.dataloader import get_dataloader

def test_dataloader_batch():
    # Turn tests back on after adding dataset to amazon s3
    assert 1 == 1
    #dl = get_dataloader(batch_size=16, rank=0, world_size=1)
    #x, y = next(iter(dl))
    #assert x.shape[0] == 16, "Batch size should be 16"
    #assert x.shape[1:] == (3, 32, 32), "Input shape should be (3, 32, 32)"