from fastapi import BackgroundTasks

from app.services.ml_runner import run_batch_inference_task


class InferenceService:
    """
    Responsible only for starting inference.

    Does NOT:
        - create batches
        - create video runs
        - parse ML output
        - save cycles
        - save logs

    It simply schedules the ML runner.
    """

    @staticmethod
    def start_batch(
        background_tasks: BackgroundTasks,
        batch_id: int,
    ) -> None:
        """
        Starts processing an entire batch in the background.
        """

        background_tasks.add_task(
            run_batch_inference_task,
            batch_id,
        )