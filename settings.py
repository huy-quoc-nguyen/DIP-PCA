import logging
import os


def get_env(env_var, default_val):
    return os.environ.get(env_var, default_val)


LOG_LEVEL_CONFIG = (lambda lvl: lvl if lvl in map(
                        logging.getLevelName,
                        [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
                    ) else logging.getLevelName(logging.INFO))(
    get_env('LOG_LEVEL_CONFIG', 'DEBUG')
)
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'json-fmt': {
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'format': '{"time": "%(asctime)s.%(msecs)03d", "level": "%(levelname)s", %(message)s, '
                      '"module": "%(module)s", "function": "%(funcName)s", '
                      '"thread": "%(thread)d", "process": "%(process)d"}'
        }
    },
    'handlers': {
        'json-log-console': {
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL_CONFIG,
            'formatter': 'json-fmt'
        }
    },
    'loggers': {
        'default': {
            'handlers': ['json-log-console'],
            'level': LOG_LEVEL_CONFIG
        }
    }
}
