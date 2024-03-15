% 实验一： 不同模型的训练与测试

python main.py --model bcresnet1 --save e1
python main.py --model bcresnet3 --save e1
python main.py --model bcresnet6 --save e1
python main.py --model bcresnet8 --save e1

python main.py --model tcresnet8 --save e1 
python main.py --model tcresnet14 --save e1

python main.py --model matchboxnet_3_1_64 --save e1 
python main.py --model matchboxnet_3_2_64 --save e1
python main.py --model matchboxnet_6_2_64 --save e1

python main.py --model convmixer --save e1

python main.py --model kwt-1 --save e1
python main.py --model kwt-2 --save e1

python main.py --model seresnet1 --save e1
python main.py --model seresnet3 --save e1
python main.py --model seresnet6 --save e1
python main.py --model seresnet8 --save e1