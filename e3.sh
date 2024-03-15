% 实验三： 不同比例的训练数据集

python main.py --model bcresnet1 --save e3 --trainratio 0.2
python main.py --model bcresnet1 --save e3 --trainratio 0.5
python main.py --model bcresnet1 --save e3 --trainratio 0.8

python main.py --model tcresnet8 --save e3 --trainratio 0.2
python main.py --model tcresnet8 --save e3 --trainratio 0.5
python main.py --model tcresnet8 --save e3 --trainratio 0.8

python main.py --model matchboxnet_3_1_64 --save e3 --trainratio 0.2
python main.py --model matchboxnet_3_1_64 --save e3 --trainratio 0.5
python main.py --model matchboxnet_3_1_64 --save e3 --trainratio 0.8

python main.py --model convmixer --save e3 --trainratio 0.2
python main.py --model convmixer --save e3 --trainratio 0.5
python main.py --model convmixer --save e3 --trainratio 0.8

python main.py --model kwt-1 --save e3 --trainratio 0.2
python main.py --model kwt-1 --save e3 --trainratio 0.5
python main.py --model kwt-1 --save e3 --trainratio 0.8

python main.py --model seresnet1 --save e3 --trainratio 0.2
python main.py --model seresnet1 --save e3 --trainratio 0.5
python main.py --model seresnet1 --save e3 --trainratio 0.8

