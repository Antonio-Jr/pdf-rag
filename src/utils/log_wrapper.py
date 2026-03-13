import functools
import inspect
import logging
import time


def get_logger(name):
    return logging.getLogger(name)


def log_execution(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        origin = func.__module__
        func_name = func.__qualname__

        start = time.perf_counter()
        logger = get_logger(origin)
        logger.info(f"🚀 [ASYNC START] {func_name}")
        try:
            result = await func(*args, **kwargs)
            logger.info(
                f"✅ [ASYNC SUCCESS] {func_name} | Time: {time.perf_counter() - start:.4f}s"
            )
            return result
        except Exception as e:
            logger.error(f"❌ [ASYNC ERROR] {func_name} | {e}")
            raise

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        origin = func.__module__
        func_name = func.__qualname__

        logger = get_logger(origin)

        def truncate(data, max_len=150):
            r = repr(data)
            return r[:max_len] + "..." if len(r) > max_len else r

        args_repr = [truncate(arg) for arg in args]
        kwargs_repr = [f"{k}={truncate(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        logger.info(f"🚀 [START] {func_name} | Args: ({signature})")

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time

            logger.info(f"✅ [SUCCESS] {func_name} | Time: {duration:.4f}s")
            return result
        except Exception as e:
            logger.error(f"❌ [ERROR] {func_name} | Fail: {str(e)}", exc_info=True)
            raise e

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
