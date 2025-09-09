import logging
from argument import get_args
from datetime import datetime, timedelta

import tools as my

if __name__ == "__main__":
    my.init_logger()
    logger = logging.getLogger(__name__)

    args = get_args()

    time0 = datetime.now()
    logger.info(f"Start with command: {args.command}.")
    if args.command == "train_dit":
        args.save_dir = my.find_ok_path(args.save_dir)
        import train_dit
        train_dit.main(args)
    elif args.command == "sample":
        import sample
        sample.main(args)
    elif args.command == "train_downstream":
        import train_downstream
        train_downstream.main(args)
    else:
        print(f"Unrecognized command: {args.command}")
    seconds = (datetime.now() - time0).total_seconds()
    logger.info(f"Used time:{timedelta(seconds=seconds)}.")