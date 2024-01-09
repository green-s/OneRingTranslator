# nllb plugin with https://opennmt.net/CTranslate2/index.html support
# author: Vladislav Janvarev

# from https://github.com/facebookresearch/fairseq/tree/nllb

from oneringcore import OneRingCore
from contextlib import contextmanager
import os
import threading

modname = os.path.basename(__file__)[:-3]  # calculating modname

model = None
tokenizers: dict = {}

model_timer = None
model_lock = threading.RLock()

cache_to_ram = False
cache_duration = 60 * 5

def reset_model_timer(cache_duration, cache_to_ram=False):
    """Reset the model timer."""
    global model_timer

    model_timer = threading.Timer(cache_duration, uncache_model, [cache_to_ram])
    model_timer.start()

def cache_model(cache_duration, cache_to_ram=False):
    """Cache the model and start or reset the timer."""
    global model, model_timer, model_lock

    with model_lock:
        if model_timer:
            model_timer.cancel()
        
        model.load_model()

        reset_model_timer(cache_duration, cache_to_ram)
        print("Model cached")

def uncache_model(cache_to_ram=False):
    """Unload the model."""
    global model, model_timer, model_lock

    with model_lock:
        if model_timer:
            model_timer.cancel()
        model_timer = None
        model.unload_model(to_cpu=cache_to_ram)
        print("Model uncached")

@contextmanager
def get_model(cache_duration, cache_to_ram=False):
    """Get the model, caching it if necessary."""
    global model, model_timer, model_lock

    with model_lock:
        if model_timer:
            model_timer.cancel()

        if not model.model_is_loaded:
            cache_model(cache_duration, cache_to_ram)
        else:
            reset_model_timer(cache_duration, cache_to_ram)

        yield model
        

# -------- lang list

langlist = [
    "en",
    "ru",
    "es",
    "de",
    "fr",
    "it",
    "pt",
    "pl",
    "nl",
    "tr",
    "vi",
    "cs",
    "id",
    "ro",
    "sv",
    "hu",
    "uk",
    "fa",
    "ja",
    "el",
    "fi",
    "zh",
    "da",
    "th",
    "no",
    "bg",
    "ko",
    "ar",
    "sk",
    "ca",
    "lt",
    "he",
    "sl",
    "et",
    "lv",
    "hi",
    "sq",
    "az",
    "hr",
    "ta",
    "ms",
    "ml",
    "sr",
    "kk",
    "te",
    "mr",
    "is",
    "bs",
    "mk",
    "gl",
    "eu",
    "bn",
    "be",
    "ka",
    "fil",
    "mn",
    "af",
    "uz",
    "gu",
    "kn",
    "kaa",
    "sw",
    "ur",
    "ne",
    "cy",
    "hy",
    "ky",
    "si",
    "tt",
    "tg",
    "la",
    "so",
    "ga",
    "km",
    "mt",
    "eo",
    "ps",
    "rw",
    "ku",
    "lo",
    "fy",
    "ha",
    "my",
    "dv",
    "pa",
    "ckb",
    "lb",
    "mg",
    "ht",
    "ug",
    "am",
    "or",
    "fo",
    "gd",
    "ba",
    "tk",
    "mi",
    "hmn",
    "grc",
    "jv",
    "ceb",
    "sd",
    "yi",
    "kaa_Latn",
    "sn",
    "co",
    "su",
    "pap",
    "ig",
    "zu",
    "xh",
    "sm",
    "ny",
    "yo",
    "cv",
    "el_Latn",
    "kl",
    "haw",
    "gsw",
    "tet",
    "st",
    "lus",
    "oc",
    "as",
    "rm",
    "br",
    "sah",
    "hi_Latn",
    "se",
    "cnh",
    "om",
    "ce",
    "udm",
    "lg",
    "os",
    "nv",
    "kha",
    "ilo",
    "ctd_Latn",
    "vec",
    "hil",
    "tyv",
    "iba",
    "ru_Latn",
    "kbd",
    "ti",
    "sa",
    "av",
    "bo",
    "zza",
    "ber_Latn",
    "otq",
    "te_Latn",
    "bua",
    "ts",
    "cfm",
    "tn",
    "krc",
    "ak",
    "meo",
    "chm",
    "to",
    "ee",
    "nso",
    "ady",
    "rom",
    "bho",
    "ltg",
    "fj",
    "yua",
    "gn",
    "az_RU",
    "ln",
    "ada",
    "myv",
    "bik",
    "tlh",
    "kbp",
    "war",
    "wa",
    "bew",
    "rcf",
    "ta_Latn",
    "kac",
    "iu",
    "ay",
    "kum",
    "qu",
    "bgp",
    "hif",
    "kw",
    "nan_Latn_TW",
    "srn",
    "tly_IR",
    "sg",
    "gom",
    "ml_Latn",
    "kj",
    "ksd",
    "dz",
    "kv",
    "msi",
    "ve",
    "zap",
    "zxx_xx_dtynoise",
    "meu",
    "iso",
    "ium",
    "nhe",
    "tyz",
    "hui",
    "new",
    "mdf",
    "pag",
    "gv",
    "gag",
    "ngu",
    "quc",
    "mam",
    "min",
    "ho",
    "pon",
    "mrj",
    "lu",
    "gom_Latn",
    "alt",
    "nzi",
    "tzo",
    "bci",
    "dtp",
    "abt",
    "bbc",
    "pck",
    "mai",
    "mps",
    "emp",
    "mgh",
    "tab",
    "crh",
    "tbz",
    "ss",
    "chk",
    "bru",
    "nnb",
    "fon",
    "ppk",
    "tiv",
    "btx",
    "bg_Latn",
    "mbt",
    "ace",
    "tvl",
    "dov",
    "ach",
    "xal",
    "cuk",
    "kos",
    "crs",
    "wo",
    "bts",
    "ubu",
    "gym",
    "ibb",
    "ape",
    "stq",
    "ang",
    "enq",
    "tsg",
    "shn",
    "kri",
    "kek",
    "rmc",
    "acf",
    "fip",
    "syr",
    "qub",
    "bm",
    "tzh",
    "jiv",
    "kn_Latn",
    "kjh",
    "yap",
    "ban",
    "tuc",
    "tcy",
    "cab",
    "cak",
    "din",
    "zh_Latn",
    "arn",
    "lrc",
    "rwo",
    "hus",
    "bum",
    "mak",
    "frp",
    "seh",
    "twu",
    "kmb",
    "ksw",
    "sja",
    "amu",
    "mad",
    "quh",
    "dyu",
    "toj",
    "ch",
    "sus",
    "nog",
    "jam",
    "gui",
    "nia",
    "mas",
    "bzj",
    "mkn",
    "lhu",
    "ctu",
    "kg",
    "inb",
    "guh",
    "rn",
    "bus",
    "mfe",
    "sda",
    "bi",
    "cr_Latn",
    "gor",
    "jac",
    "chr",
    "mh",
    "mni",
    "wal",
    "teo",
    "gub",
    "qvi",
    "tdx",
    "rki",
    "djk",
    "nr",
    "zne",
    "izz",
    "noa",
    "bqc",
    "srm",
    "niq",
    "bas",
    "dwr",
    "guc",
    "jvn",
    "hvn",
    "sxn",
    "koi",
    "alz",
    "nyu",
    "bn_Latn",
    "suz",
    "pau",
    "nij",
    "sat_Latn",
    "gu_Latn",
    "msm",
    "maz",
    "qxr",
    "shp",
    "hne",
    "ktu",
    "laj",
    "pis",
    "mag",
    "gbm",
    "tzj",
    "oj",
    "ndc_ZW",
    "tks",
    "awa",
    "gvl",
    "knj",
    "spp",
    "mqy",
    "tca",
    "cce",
    "skr",
    "kmz_Latn",
    "dje",
    "gof",
    "agr",
    "qvz",
    "adh",
    "quf",
    "kjg",
    "tsc",
    "ber",
    "ify",
    "cbk",
    "quy",
    "ahk",
    "cac",
    "akb",
    "nut",
    "ffm",
    "taj",
    "ms_Arab",
    "brx",
    "ann",
    "qup",
    "ms_Arab_BN",
    "miq",
    "msb",
    "bim",
    "raj",
    "kwi",
    "tll",
    "trp",
    "smt",
    "mrw",
    "dln",
    "qvc",
    "doi",
    "ff",
]

cuda_opt = -1
to_device = "cpu"


# start function
def start(core: OneRingCore):
    manifest = {  # plugin settings
        "name": "NLLB Translate with CTranslate2 (MADLAD)",  # name
        "version": "2.0",  # version
        "translate": {
            "fb_nllb_ctranslate2_madlad": (
                init,
                translate,
            )  # 1 function - init, 2 - translate
        },
        "default_options": {
            "model": "SoybeanMilk/madlad400-10b-mt-ct2-int8_float16",  # key model
            "cuda": -1,  # -1 if you want run on CPU, 0 - if on CUDA
            "cache_duration": cache_duration,
            "cache_to_ram": False,
        },
    }
    return manifest


def start_with_options(core: OneRingCore, manifest: dict):
    global cuda_opt
    global to_device
    cuda_opt = manifest["options"].get("cuda")
    if cuda_opt == -1:
        to_device = "cpu"
    else:
        to_device = "cuda".format(cuda_opt)
    pass


def init(core: OneRingCore):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import ctranslate2

    global model, cache_duration, cache_to_ram

    # print(to_device)
    # model = AutoModelForSeq2SeqLM.from_pretrained(core.plugin_options(modname).get("model")).to(to_device)
    model = ctranslate2.Translator(
        core.plugin_options(modname).get("model"), device=to_device
    )
    model.unload_model()

    cache_duration = core.plugin_options(modname).get("cache_duration")
    cache_to_ram = core.plugin_options(modname).get("cache_to_ram")

    pass


def convert_lang(input_lang: str) -> str:
    if len(input_lang) == 2 or len(input_lang) == 3:
        # if input_lang == "en" or input_lang == "eng":
        #     return "eng_Latn"
        # if input_lang == "ru":
        #     return "rus_Cyrl"

        for lang in langlist:
            if lang.startswith(input_lang):
                return lang

    else:
        return input_lang


def translate(
    core: OneRingCore,
    text: str,
    from_lang: str = "",
    to_lang: str = "",
    add_params: str = "",
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from_lang_tr = from_lang
    to_lang_tr = to_lang
    if tokenizers.get(from_lang_tr) is None:
        tokenizers[from_lang_tr] = AutoTokenizer.from_pretrained(
            core.plugin_options(modname).get("model"), src_lang=from_lang_tr
        )

    tokenizer_from = tokenizers.get(from_lang_tr)

    prefixed_text = f"<2{to_lang_tr}> {text}"
    source = tokenizer_from.convert_ids_to_tokens(tokenizer_from.encode(prefixed_text))

    # target_prefix = [to_lang_tr]
    # results = model.translate_batch([source], target_prefix=[target_prefix])
    with get_model(cache_duration) as model:
        results = model.translate_batch([source])
    # translated_tokens = results[0].hypotheses[0][1:]
    translated_tokens = results[0].hypotheses[0]

    res = tokenizer_from.decode(tokenizer_from.convert_tokens_to_ids(translated_tokens))

    return res
