try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except Exception as e:
    print("Please install NVIDIA DALI for boosting training speed.")

import math
import time

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               # random_aspect_ratio=[0.8, 1.25],
                                               # random_area=[0.1, 1.0],
                                               random_aspect_ratio=[0.75, 1.333333],
                                               random_area=[0.08, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


if __name__ == '__main__':
    batch_size = 32
    workers = 4
    local_rank = 0
    traindir = "./imagenet1000/train"
    valdir = "./imagenet1000/val"
    crop_size = 224
    val_size = 256
    dali_cpu = False # dali_cpu = True 45s; dali_cpu = False, 18s.
    world_size = 4

    pipe_train = create_dali_pipeline(batch_size=batch_size,
                                num_threads=workers,
                                device_id=local_rank,
                                seed=12 + local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=dali_cpu,
                                shard_id=local_rank,
                                num_shards=world_size,
                                is_training=True)
    pipe_train.build()
    train_loader = DALIClassificationIterator(pipe_train, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)


    pipe_val = create_dali_pipeline(batch_size=batch_size,
                                num_threads=workers,
                                device_id=local_rank,
                                seed=12 + local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=dali_cpu,
                                shard_id=local_rank,
                                num_shards=world_size,
                                is_training=False)
    pipe_val.build()
    val_loader = DALIClassificationIterator(pipe_val, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    start = time.time()
    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(val_loader._size / batch_size))
        print(i,input.size(),input.is_cuda,target.size(),target.is_cuda,train_loader_len,len(val_loader))
    end = time.time()
    print(end-start)