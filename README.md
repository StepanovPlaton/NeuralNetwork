# Neural Network ++

> Neural Network++ - это движок для ~путешествия в Мордор~ создания нейронных сетей

## Стек:

- [C++ 23](https://ru.wikipedia.org/wiki/C%2B%2B23)
- [OpenCL](https://ru.wikipedia.org/wiki/OpenCL) - библиотека GPGPU (вычисления на видеокарте)
- [pybind11](https://github.com/pybind/pybind11) - создание С++ библиотеки для Python
- **Всё!** :wink:

## О проекте:

- Движок для создания нейронных сетей
- Поддержка вычислений [на CPU](./src/tensor/cpu/) или [на GPU с использованием OpenCL](./src/tensor/opencl/)
  - [Алгоритмы массового параллелизма на GPU](./src/tensor/opencl/kernels) для быстрых вычислений
  - Классические алгоритмы на CPU для возможности проверки
- [Класс Tensor](./src/tensor/tensor.hpp) для работы с тензорами произвольной размерности

## Forward & Back propogation - это путешествие в Мордор и обратно!

![](back_propogation.png)

> Верная смерть. Никаких шансов на успех. Так чего же мы ждём?!

### Над проектом работали [StepanovPlaton](https://github.com/StepanovPlaton) и [Fluorouacil](https://github.com/Fluorouacil)!
