# 实验四 不同信噪比下的


python main.py --model bcresnet1 --save e4.1 --noise 10
python main.py --model bcresnet1 --save e4.2 --noise 0
python main.py --model bcresnet1 --save e4.3 --noise -10

python main.py --model tcresnet8 --save e4.1 --noise 10
python main.py --model tcresnet8 --save e4.2 --noise 0
python main.py --model tcresnet8 --save e4.3 --noise -10

python main.py --model matchboxnet_3_1_64 --save e4.1 --noise 10
python main.py --model matchboxnet_3_1_64 --save e4.2 --noise 0
python main.py --model matchboxnet_3_1_64 --save e4.3 --noise -10

python main.py --model convmixer --save e4.1 --noise 10
python main.py --model convmixer --save e4.2 --noise 0
python main.py --model convmixer --save e4.3 --noise -10

python main.py --model kwt-1 --save e4.1 --noise 10
python main.py --model kwt-1 --save e4.2 --noise 0
python main.py --model kwt-1 --save e4.3 --noise -10

python main.py --model seresnet1 --save e4.1 --noise 10
python main.py --model seresnet1 --save e4.2 --noise 0
python main.py --model seresnet1 --save e4.3 --noise -10
```

```bash

