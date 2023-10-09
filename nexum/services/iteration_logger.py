from time import time
import inspect

from tqdm.auto import tqdm


class DataStorage(object):
    def __getattr__(self, name):
        try:
            return super().__getattr__(self, name)
        except Exception:
            return None

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)


class IterationLogger(object):
    modules_separator = ", "
    desc = ""
    # bar_format = "{desc} |{bar}| [{elapsed}; {remaining}] {postfix}"
    bar_format = "{desc} |{bar}|{postfix}"

    def __init__(self, modules=None):
        default_modules = {"time": self.took_time}
        self.modules = modules or default_modules
        self.ds = DataStorage()

    def __len__(self):
        return self._n

    def __iter__(self):
        return self

    @staticmethod
    def took_time(iteration, total, took_time):
        s = int(took_time)
        us = int((took_time * 100) % 100)
        if iteration != total:
            return f"ETA: {s}s {us}us"
        else:
            return f"{s}s {us}us/sample"

    def __next__(self):
        self.ds.took_time = time() - self.time_start
        self.ds.iteration = self.i
        self.ds.total = self.tqdm.total

        self.tqdm.set_postfix_str(self.collect_postfix_str())
        self.tqdm.set_description_str(
            f"{self.desc}{self.i:{len(str(self.tqdm.total))}d}/{self.tqdm.total}"
        )

        self.i += 1

        # self.time_start = time()
        return next(self._iterator)

    def __call__(self, iterator, *args, **kwargs):
        number_of_iterations = None

        if hasattr(iterator, "__len__"):
            number_of_iterations = len(iterator)

        self._iterator = iter(iterator)
        self._n = number_of_iterations
        self.i = 0
        self.tqdm = tqdm(self, bar_format=self.bar_format, *args, **kwargs)
        self.time_start = time()
        return self.tqdm

    def collect_postfix_str(self):
        modules = []

        for module_name in self.modules:
            module_factory = self.modules[module_name]
            spec = inspect.getfullargspec(module_factory)
            module_data = dict()

            for param_name in spec[0]:
                module_data[param_name] = self.ds[param_name]

            try:
                modules.append(module_factory(**module_data))
            except:
                ...

        modules_string = self.modules_separator.join(modules)
        return modules_string
