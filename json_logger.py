import json
import logging


def _json_wrapper(log_method):
    def _dict_to_embeded_json(dict_data: dict) -> str:
        # Remove '{' and '}' in json message and embed the string message to JSON logs
        return json.dumps(dict_data)[1: -1]

    def _convert_log_method(level, msg, args, exc_info=None, extra=None, stack_info=False):
        if isinstance(msg, dict):
            json_msg = _dict_to_embeded_json(msg)
            return log_method(level, json_msg, args, exc_info, extra, stack_info)
        elif isinstance(msg, str):
            json_msg = _dict_to_embeded_json({'msg': msg})
            return log_method(level, json_msg, args, exc_info, extra, stack_info)
        else:
            json_msg = _dict_to_embeded_json({'msg': str(msg)})
            return log_method(level, json_msg, args, exc_info, extra, stack_info)

    return _convert_log_method


class JsonLogger(logging.Logger):
    def __getattribute__(self, item):
        if item == '_log':
            return _json_wrapper(super(JsonLogger, self).__getattribute__(item))
        else:
            return super(JsonLogger, self).__getattribute__(item)
