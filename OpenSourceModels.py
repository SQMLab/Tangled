import os
import pandas as pd
import argparse
import logging
from modules.Untangler import OpenUntangler

logger = logging.getLogger(__name__)
job_id, model_id, start, end, rq = None, None, None, None, None

def parse_arguments():
    global job_id, model_id, start, end, rq

    parser = argparse.ArgumentParser(description="Run Open LLM-based untangling")
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="Job ID"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Full name of the LLM"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        required=True,
        help="Index of First Samples"
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Index of Last Samples"
    )
    parser.add_argument(
        "--rq",
        type=int,
        required=True,
        choices=[11, 12, 21, 22, 23],
        help="Full name of the LLM"
    )
    args = parser.parse_args()
    model_id = args.name
    job_id = args.job_id
    start = args.start
    end = args.end
    rq = args.rq

def setup_logging():
    global job_id, model_id, start, end, rq
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename="./logs/" + str(job_id) + "." + model_id.split("/")[1] + "." + str(start) + "-" + str(end) + "." + str(rq) + ".txt",
                                filemode='w',
                                format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.INFO)

if __name__ == "__main__":
    parse_arguments()
    setup_logging()

    batch_size = 2
    try:
        config_map = {
            11: ("./Results/RQ1", "WithoutMsg", (model_id, False, 0, False, batch_size, logger)),
            12: ("./Results/RQ1", "WithMsg", (model_id, True, 0, False, batch_size, logger)),
            21: ("./Results/RQ2", "FewShot", (model_id, True, 2, False, batch_size, logger)),
            22: ("./Results/RQ2", "COT", (model_id, True, 0, True, batch_size, logger)),
            23: ("./Results/RQ2", "FewShotCOT", (model_id, True, 2, True, batch_size, logger)),
        }

        logger.info("Job Starting.")
        logger.info(f"Job Id: {job_id}, Model: {model_id}, Start: {start}, End: {end}, RQ: {config_map[rq][1]}")

        logger.info("Loading dataset.")
        df = pd.read_csv("./data/Complete_GoldSet.csv")
        df = df[start:end]

        
        logger.info("Setting variables and creating result folders.")
        if rq in config_map:
            save_path, folder, args = config_map[rq]
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, folder)
            os.makedirs(save_path, exist_ok=True)

            untangler = OpenUntangler(*args)

        model_name = model_id.replace("/", "-")
        csv_file_path = f"{save_path}/{model_name}.csv"

        logger.info("Starting detection.")
        result = untangler.batch_detect(df)
        logger.info("Detection complete.")

        if os.path.exists(csv_file_path):
            logger.info("Loading old result file to merge")
            df = pd.read_csv(csv_file_path)
            result = pd.concat([df, result])
            logger.info("Merge complete")

        logger.info("Saving results.")
        result.to_csv(csv_file_path, index = False)

        logger.info("Job Successful.")
    except Exception as e:
        logger.error("Job Failed.")
        logger.exception("Exception occurred", exc_info=True)

        raise e

