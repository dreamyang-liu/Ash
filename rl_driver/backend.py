"""HTTP only: execution state and retries belong to Run Store workers."""

from urllib.parse import quote

from runstore.client import Client


# Long agent sessions contain cumulative session-state events.  Fetching the
# Run Store maximum of 1,000 events in one response can make a single page
# hundreds of MiB and exceed the HTTP client's timeout even though the job has
# already succeeded.  Bound response size while retaining cursor pagination.
EVENT_PAGE_SIZE = 50


def segment(value: str) -> str:
    return quote(value, safe="")


class RunStoreClient(Client):
    def all_events(self, job_id: str, *, attempt_id: str) -> list[dict]:
        result, after = [], 0
        while True:
            page = self.events(
                job_id,
                attempt_id=attempt_id,
                after=after,
                limit=EVENT_PAGE_SIZE,
            )
            if not page:
                return result
            cursor = page[-1]["seq"]
            if cursor <= after:
                raise ValueError("Run Store event cursor did not advance")
            result.extend(page)
            after = cursor

    def events_of_type(
        self,
        job_id: str,
        *,
        attempt_id: str,
        event_types: tuple[str, ...],
        limit: int = 1000,
        newest: bool = False,
    ) -> list[dict]:
        return self.events(
            job_id,
            attempt_id=attempt_id,
            event_types=event_types,
            limit=limit,
            newest=newest,
        )

    def request_cancel(self, job_id: str) -> bool:
        response = self.http.post(f"/v1/jobs/{segment(job_id)}/cancel")
        response.raise_for_status()
        return True

    cancel_queued = request_cancel

    def tools(self, job_id: str, *, attempt_id: str, after: int = 0) -> list[dict]:
        response = self.http.get(f"/v1/jobs/{segment(job_id)}/tools", params={
            "attempt_id": attempt_id, "after": after,
        })
        response.raise_for_status()
        return response.json()

    def points(self, job_id: str, *, attempt_id: str) -> list[dict]:
        response = self.http.get(f"/v1/jobs/{segment(job_id)}/recovery-points",
                                 params={"attempt_id": attempt_id})
        response.raise_for_status()
        return response.json()

    def all_tools(self, job_id: str, *, attempt_id: str) -> list[dict]:
        result = []
        after = 0
        while True:
            page = self.tools(job_id, attempt_id=attempt_id, after=after)
            if not page:
                return result
            cursor = page[-1]["depth"]
            if cursor <= after:
                raise ValueError("Run Store tool cursor did not advance")
            result.extend(page)
            after = cursor
