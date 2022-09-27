import sys
import time
import itertools

import pandas as pd
import multiprocessing
from typing import Callable, Tuple, Union
from multiprocessing.pool import Pool as sec_pool


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class AsyncMP:

    """
    This class is used to compute task in parallel using multiprocessing.
    """
    DEFAULT_NCPU = multiprocessing.cpu_count() - 1

    def __init__(self, **kwargs):

        self._num_cpu = kwargs.get('num_cpu', multiprocessing.cpu_count() - 1)

    def exec(self, params: list, func: Callable[[Tuple[str, pd.DataFrame]], Union[pd.DataFrame, pd.Series]],
             logger: Callable[[str], None] = sys.stdout) -> pd.DataFrame:

        start = time.time()
        # logger.flush()
        # logger.write("\nUsing {} CPUs in parallel...\n".format(self._num_cpu))

        with multiprocessing.Pool(self._num_cpu) as pool:
            result = pool.starmap_async(func, params)
            # cycler = itertools.cycle('\|/―')
            # while not result.ready():
            #     value = "\rTasks left: {} / {}. {}\t".format(
            #         result._number_left, len(params),
            #         next(cycler))
            #     logger.write(value)
            #     logger.flush()
            #     time.sleep(0.1)
            got = result.get()
        logger.write("\nTasks completed. Processed {} df in {:.1f}s\n".format(
            len(got), time.time() - start))

        return got

    def exec_function_in_parallel(self, tab_parameter: list,
                                  func,
                                  logger: Callable[[str], None] = sys.stdout
                                  ) -> pd.DataFrame:

        start = time.time()
        logger.flush()
        logger.write("\nUsing {} CPUs in parallel...\n".format(self._num_cpu))

        with MyPool(self._num_cpu) as pool:
            result = pool.starmap(func, tab_parameter, chunksize=1)

        return result

    def exec_function_with_pool_map(self, tab_parameter: list,
                                  func,
                                  logger: Callable[[str], None] = sys.stdout
                                  ) -> pd.DataFrame:
        start = time.time()
        logger.flush()
        logger.write("\nUsing {} CPUs in parallel...\n".format(self._num_cpu))
        pool = MyPool(self._num_cpu)
        results = pool.starmap(func, tab_parameter)
        pool.close()
        pool.join()

        return results

    @staticmethod
    def exec_function_in_slavequeue(queue, func, *args):

        queue.put(func(*args))
        queue.put(None)
