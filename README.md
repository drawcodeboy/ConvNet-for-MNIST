# ConvNet-for-MNIST
## Description
* 나는 코드를 짤 수 있는 사람일까
* 프로젝트 기간: 24/06/06 15:35~18:59 (3시간 24분)
## Usage
### Download Repository
```
git clone https://github.com/drawcodeboy/ConvNet-for-MNIST.git
```
### Virtual Environment
```
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```
### Train
```
python main.py --mode=train --epochs=1 --lr=1e-4 --batch_size=16 --device=cpu --file_name=trained_1epoch_convnet.pt
```
### Test
```
python main.py --mode=inference --device=cpu --file_name=trained_1epoch_convnet.pt --index=0
```