from time import perf_counter


def perfcall(func, *args, **kwargs):
    start = perf_counter()

    value = func(*args, **kwargs)

    print(f'{func}: {perf_counter() - start}')
    return value
