"""Thread-safe, out-of-band termination for a single rollout."""

import threading


class RunAborted(RuntimeError):
    pass


class RunControl:
    """A safety failure must reach the driver, not rely on model cooperation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reason = None
        self._stop_reason = None
        self._callbacks = []

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def stop_reason(self):
        with self._lock:
            return self._stop_reason

    def request_stop(self, reason, *, stop_reason=None):
        with self._lock:
            if self._reason is not None:
                return
            self._reason = reason
            self._stop_reason = stop_reason
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except RuntimeError:
                pass  # a driver that has already closed its loop is stopped

    def subscribe(self, callback):
        with self._lock:
            self._callbacks.append(callback)
            already_stopped = self._reason is not None
        if already_stopped:
            callback()

        def unsubscribe():
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)
        return unsubscribe

    def raise_if_stopped(self):
        if self.reason is not None:
            raise RunAborted(self.reason)
