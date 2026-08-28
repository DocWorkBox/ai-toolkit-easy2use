from toolkit.extension import Extension


class AceStepCaptionerExtension(Extension):
    uid = "AceStepCaptioner"
    name = "Ace Step Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .AceStepCaptioner import AceStepCaptioner

        return AceStepCaptioner


class Qwen3VLCaptionerExtension(Extension):
    uid = "Qwen3VLCaptioner"
    name = "Qwen 3VL Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3VLCaptioner import Qwen3VLCaptioner

        return Qwen3VLCaptioner


class RemoteAPICaptionerExtension(Extension):
    uid = "RemoteAPICaptioner"
    name = "Remote API Captioner"

    @classmethod
    def get_process(cls):
        from .RemoteAPICaptioner import RemoteAPICaptioner

        return RemoteAPICaptioner


class Qwen3OmniCaptionerExtension(Extension):
    uid = "Qwen3OmniCaptioner"
    name = "Qwen 3 Omni Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3OmniCaptioner import Qwen3OmniCaptioner

        return Qwen3OmniCaptioner


class Qwen2_5OmniH3CaptionerExtension(Extension):
    uid = "Qwen2_5OmniH3Captioner"
    name = "Qwen2.5 Omni H3 Captioner"

    @classmethod
    def get_process(cls):
        from .Qwen2_5OmniH3Captioner import Qwen2_5OmniH3Captioner

        return Qwen2_5OmniH3Captioner


class Ideogram4CaptionerExtension(Extension):
    uid = "Ideogram4Captioner"
    name = "Ideogram4 Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Ideogram4Captioner import Ideogram4Captioner

        return Ideogram4Captioner


class Ideogram4APICaptionerExtension(Extension):
    uid = "Ideogram4APICaptioner"
    name = "Ideogram4 API Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Ideogram4APICaptioner import Ideogram4APICaptioner

        return Ideogram4APICaptioner


AI_TOOLKIT_EXTENSIONS = [
    AceStepCaptionerExtension,
    Qwen3VLCaptionerExtension,
    RemoteAPICaptionerExtension,
    Qwen2_5OmniH3CaptionerExtension,
    Qwen3OmniCaptionerExtension,
    Ideogram4CaptionerExtension,
    Ideogram4APICaptionerExtension,
]
