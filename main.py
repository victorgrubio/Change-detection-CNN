import argparse
import logging
import traceback
from datetime import datetime
from setup_logging import setup_logging
import pyximport
pyximport.install()
from utils import model_preprocessing as model_prep
from change_detection import ChangeDetection

if __name__ == '__main__':
    # Input arguments and options
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model',
                        default='models_segmentation_cd/25_09-11_22/')
    parser.add_argument('--mode', type=str, required=True,
                        help='specify video, img_folder or img mode')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode - to be determined')
    parser.add_argument('--align', help='disable alignment to images',
                        action='store_true', default=False)
    parser.add_argument('--display',
                        help='images are not shown during prediction',
                        action='store_true')
    parser.add_argument('--save_results', '-s',
                        help='save video or image result of prediction',
                        action='store_true')
    parser.add_argument('--fore_path', type=str,
                        help='Path of foreground content')
    parser.add_argument('--back_path', type=str,
                        help='Path of background content')
    args = vars(parser.parse_args())
    model = model_prep.load_model(args['model'])
    date = "{:%d_%m-%H_%M}".format(datetime.now())
    logger = logging.getLogger(__name__)
    setup_logging()
    change_detection = ChangeDetection(logger, args, model)
    try:
        if args['mode'] == 'image':
            change_detection.predict_img(args['fore_path'], args['back_path'])
        elif args['mode'] == 'image_folder':
            change_detection.predict_folder(
                args['fore_path'], args['back_path'])
        elif args['mode'] == 'video':
            change_detection.predict_video(
                args['fore_path'], args['back_path'])
        else:
            logger.error('Mode must be image, image_folder or video')
            raise AttributeError('Mode must be image, image_folder or video')
    # CTRL+C handler
    except KeyboardInterrupt:
        # Exit video recording after analysis has been completed
        logger.info('User has interrupted the program ...')
        change_detection.finalize()
    except Exception:
        logger.error('Exiting due to Exception...')
        logger.error(traceback.format_exc())
        change_detection.finalize_video_queues()
