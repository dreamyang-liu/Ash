"""A synchronous HTTP call whose async transport is cancelled and closed on stop."""

import asyncio

import httpx

from harness.core.control import RunAborted, RunControl


def post(url: str, *, timeout_s: float, control: RunControl | None = None, **kwargs) -> httpx.Response:
    async def request():
        if control:
            control.raise_if_stopped()
        loop = asyncio.get_running_loop()
        owner = asyncio.current_task()
        unsubscribe = (control.subscribe(lambda: loop.call_soon_threadsafe(owner.cancel))
                       if control else lambda: None)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                return await asyncio.wait_for(client.post(url, **kwargs), timeout_s)
        except asyncio.CancelledError:
            if control and control.reason:
                raise RunAborted(control.reason) from None
            raise
        finally:
            unsubscribe()

    return asyncio.run(request())
