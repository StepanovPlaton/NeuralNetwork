# Neural Network ++

> Neural Network++ - это движок для ~путешествия в Мордор~ создания нейронных сетей написанный на С++

## Стек:

- [C++ 23](https://ru.wikipedia.org/wiki/C%2B%2B23)
- [OpenCL](https://ru.wikipedia.org/wiki/OpenCL)
- **Всё!** :wink:

## О проекте:

- Движок для создания нейронных сетей
- Поддерка вычислений [на CPU](./math/tensor/cpu) или [на GPU](./math/tensor/cpu)
  - Полиморные пространства имён CPU и GPU соответственно
  - [Алгоритмы с массовым параллелизмом на GPU](./kernels) для ускорения
  - Классические алгоритмы на CPU для проверки
- [Класс Tensor](./math/tensor/tensor.hpp) для работы с тензорами N-ой размерности и [классы Scalar, Vector, Matrix и Tensor3](./math/tensor/tensor.hpp) с размерно-специфичной логикой
- [Классы ScalarMath, VectorMath, MatrixMath, Tensor3Math](./math/tensor/math.hpp) с базовыми математическими функциями

## Запуск:

- **Windows:**
  ```
    make run
  ```

## Forward & Back propogation - это путешествие в Мордор и обратно!

![back_propogation.png]()

> Верная смерть. Никаких шансов на успех. Так чего же мы ждём?!

### Над проектом работали [StepanovPlaton](https://github.com/StepanovPlaton) и [Fluorouacil](https://github.com/Fluorouacil)!
