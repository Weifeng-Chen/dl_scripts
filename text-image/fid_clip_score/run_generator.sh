#!/bin/bash
ps aux | grep -E 'run_watch.sh|watch.py' |awk '{print $2}' | xargs kill -9 # kill previous watchdog
guidance_scales=(1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
for i in {0..7}
do
    echo ${i}
    CUDA_VISIBLE_DEVICES=${i} nohup python coco_sample_generator.py --guidance_scale ${guidance_scales[${i}]} --batch_size 16 --sample_step 20 > stable_generator.log 2>&1 &
done
wait
bash ~/release_watchdog.sh # start watchdog