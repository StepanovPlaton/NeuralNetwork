# Neural Network ++

> Neural Network++ - это движок для ~путешествия в Мордор~ создания нейронных сетей написанный на С++

## Стек:

- [C++ 23](https://ru.wikipedia.org/wiki/C%2B%2B23)
- [OpenCL](https://ru.wikipedia.org/wiki/OpenCL)
- [pybind11](https://github.com/pybind/pybind11)
- **Всё!** :wink:

## О проекте:

- Движок для создания нейронных сетей
- Поддерка вычислений на CPU или на GPU
  - [Алгоритмы с массовым параллелизмом на GPU](./src/tensor/opencl/kernels) для ускорения
  - Классические алгоритмы на CPU для проверки вычислений
- [Класс Tensor](./src/tensor/tensor.hpp) для работы с тензорами произвольной размерности

## Forward & Back propogation - это путешествие в Мордор и обратно!

![](back_propogation.png)

> Верная смерть. Никаких шансов на успех. Так чего же мы ждём?!

### Над проектом работали [StepanovPlaton](https://github.com/StepanovPlaton) и [Fluorouacil](https://github.com/Fluorouacil)!
