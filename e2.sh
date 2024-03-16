实验二 结合非关键词音频和无声音频

python main.py --model bcresnet1 --save e2 --unknown_ratio 1.0
python main.py --model bcresnet3 --save e2 --unknown_ratio 1.0
python main.py --model bcresnet6 --save e2 --unknown_ratio 1.0
python main.py --model bcresnet8 --save e2 --unknown_ratio 1.0

python main.py --model tcresnet8 --save e2 --unknown_ratio 1.0
python main.py --model tcresnet14 --save e2 --unknown_ratio 1.0

python main.py --model matchboxnet_3_1_64 --save e2  --unknown_ratio 1.0
python main.py --model matchboxnet_3_2_64 --save e2 --unknown_ratio 1.0
python main.py --model matchboxnet_6_2_64 --save e2 --unknown_ratio 1.0

python main.py --model convmixer --save e2 --unknown_ratio 1.0

python main.py --model kwt-1 --save e2 --unknown_ratio 1.0
python main.py --model kwt-2 --save e2 --unknown_ratio 1.0

python main.py --model seresnet1 --save e2 --unknown_ratio 1.0
python main.py --model seresnet3 --save e2 --unknown_ratio 1.0
python main.py --model seresnet6 --save e2 --unknown_ratio 1.0
python main.py --model seresnet8 --save e2 --unknown_ratio 1.0