from ast import literal_eval
import distutils

def _decode_cfg_value(v):
	if not isinstance(v, str):
		return v
	try:
		v = literal_eval(v)
	except ValueError:
		pass
	except SyntaxError:
		pass
	return

def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def parse_argdict(pretrain_settings):
    assert len(pretrain_settings) % 2 == 0, "Missing args in pretrain_settings"
    parsed_pretrain_settings = {}
    for full_key, v in zip(pretrain_settings[0::2], pretrain_settings[1::2]):
        try:
            v = bool(distutils.util.strtobool(v))
        except:
            try:
                v = int(v)
            except:
                pass

        parsed_pretrain_settings[full_key] = v
    return parsed_pretrain_settings