from tensor.tensor import *


def test_matrix_operations():
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ БИБЛИОТЕКИ MATRIX")
    print("=" * 50)

    # Тест создания матриц
    print("\n1. СОЗДАНИЕ МАТРИЦ:")
    print("-" * 30)

    # Создание матрицы с заполнением одним значением
    m1 = Matrix([2, 3], 1.0)
    print(f"Matrix([2, 3], 1.0) = {m1}")

    # Создание матрицы с разными значениями
    m2 = Matrix([2, 3], 2.0, 3.0)
    print(f"Matrix([2, 3], 2.0, 3.0) = {m2}")

    # Создание матрицы для умножения
    m3 = Matrix([3, 2], 2.0)
    print(f"Matrix([3, 2], 2.0) = {m3}")

    # Тест получения свойств
    print("\n2. СВОЙСТВА МАТРИЦ:")
    print("-" * 30)
    print(f"m1.get_shape() = {m1.get_shape()}")
    print(f"m1.get_axes() = {m1.get_axes()}")
    print(f"m1.get_size() = {m1.get_size()}")

    # Тест доступа к элементам
    print("\n3. ДОСТУП К ЭЛЕМЕНТАМ:")
    print("-" * 30)
    print(f"m1[0] = {m1[0]}")
    print(f"m1[0, 1] = {m1[0, 1]}")

    # Установка значений
    m1[0, 1] = 5.0
    print(f"После m1[0, 1] = 5.0: {m1}")

    # Тест арифметических операций
    print("\n4. АРИФМЕТИЧЕСКИЕ ОПЕРАЦИИ:")
    print("-" * 30)

    # Сложение
    m_add = m1 + m2
    print(f"m1 + m2 = {m_add}")

    # Вычитание
    m_sub = m1 - m2
    print(f"m1 - m2 = {m_sub}")

    # Умножение на скаляр
    m_mul_scalar = m1 * 2.0
    print(f"m1 * 2.0 = {m_mul_scalar}")

    # Поэлементное умножение
    m_mul_element = m1 * m2
    print(f"m1 * m2 (поэлементно) = {m_mul_element}")

    # Деление на скаляр
    m_div = m1 / 2.0
    print(f"m1 / 2.0 = {m_div}")

    # Унарные операторы
    m_neg = -m1
    print(f"-m1 = {m_neg}")
    m_pos = +m1
    print(f"+m1 = {m_pos}")

    # Тест матричного умножения
    print("\n5. МАТРИЧНОЕ УМНОЖЕНИЕ:")
    print("-" * 30)
    try:
        m_matmul = m1 @ m3
        print(f"m1 @ m3 = {m_matmul}")
    except Exception as e:
        print(f"Ошибка при матричном умножении: {e}")

    # Тест транспонирования
    print("\n6. ТРАНСПОНИРОВАНИЕ:")
    print("-" * 30)
    m_transposed = m1.t()
    print(f"m1.t() = {m_transposed}")

    try:
        m_transpose_method = m1.transpose(0, 1)
        print(f"m1.transpose(0, 1) = {m_transpose_method}")
    except Exception as e:
        print(f"Ошибка при transpose(0, 1): {e}")

    try:
        m_transpose_list = m1.transpose([0, 1])
        print(f"m1.transpose([0, 1]) = {m_transpose_list}")
    except Exception as e:
        print(f"Ошибка при transpose([0, 1]): {e}")

    # Тест операций на месте
    print("\n7. ОПЕРАЦИИ НА МЕСТЕ:")
    print("-" * 30)

    m_test = Matrix([2, 2], 1.0)
    print(f"Исходная матрица: {m_test}")

    m_test += 2.0
    print(f"После m_test += 2.0: {m_test}")

    m_test -= 1.0
    print(f"После m_test -= 1.0: {m_test}")

    m_test *= 3.0
    print(f"После m_test *= 3.0: {m_test}")

    m_test /= 2.0
    print(f"После m_test /= 2.0: {m_test}")

    # Тест с вашими матрицами из примера
    print("\n8. ТЕСТ С ВАШИМИ МАТРИЦАМИ:")
    print("-" * 30)

    a = Matrix([2, 3], 2)
    b = Matrix([3, 2], 1)

    print(f"a = {a}")
    print(f"b = {b}")

    try:
        result = a @ b
        print(f"a @ b = {result}")
    except Exception as e:
        print(f"Ошибка при a @ b: {e}")

    # Тест обратных операций
    print("\n9. ОБРАТНЫЕ ОПЕРАЦИИ:")
    print("-" * 30)

    m_base = Matrix([2, 2], 3.0)
    print(f"Исходная матрица: {m_base}")

    # Правое сложение
    m_radd = 2.0 + m_base
    print(f"2.0 + m_base = {m_radd}")

    # Правое умножение
    m_rmul = 2.0 * m_base
    print(f"2.0 * m_base = {m_rmul}")

    # Правое вычитание
    m_rsub = 10.0 - m_base
    print(f"10.0 - m_base = {m_rsub}")

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 50)


def test_edge_cases():
    print("\n\n10. ТЕСТ ГРАНИЧНЫХ СЛУЧАЕВ:")
    print("=" * 50)

    try:
        # Попытка создания с разными параметрами
        m_empty = Matrix([0, 0])
        print(f"Matrix([0, 0]) = {m_empty}")
    except Exception as e:
        print(f"Ошибка при создании Matrix([0, 0]): {e}")

    try:
        # Попытка доступа к несуществующему элементу
        m_test = Matrix([2, 2], 1.0)
        print(f"Попытка доступа к m_test[5, 5]: ", end="")
        value = m_test[5, 5]
        print(value)
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    test_matrix_operations()
    test_edge_cases()
