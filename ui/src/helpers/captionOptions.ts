import { GroupedSelectOption, SelectOption } from "@/types";
import {
    buildCaptionPrompt,
    defaultCaptionPromptTemplate,
    defaultCaptionTargetLanguage,
} from '@/helpers/captionPrompts';

type CaptionGroup = 'image' | 'music';
type AdditionalSections =
    'caption.model_name_or_path2'
    | 'caption.caption_prompt'
    | 'caption.max_res'
    | 'caption.max_new_tokens'
    | 'caption.fixed_caption'
    | 'caption.api_concurrency'
    | 'caption.api_base_url'
    | 'caption.api_key'
    | 'caption.api_protocol'
    | 'caption.prompt_template'
    | 'caption.target_lang';

export interface CaptionOption {
    name: string;
    label: string;
    group: CaptionGroup;
    hasMultiLinePrompts?: boolean;
    supportsQuantization?: boolean;
    supportsLowVram?: boolean;
    defaults?: { [key: string]: any };
    additionalSections?: AdditionalSections[];
    name_or_path_options?: SelectOption[];
    name_or_path2_options?: SelectOption[];
}

const defaultNameOrPath = '';

const extensionsAudio = ['mp3', 'wav', 'flac', 'ogg'];
const extensionsImage = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];

const defaultExtensions = [...extensionsImage];

const defaultImageCaptionPrompt = buildCaptionPrompt(defaultCaptionPromptTemplate, defaultCaptionTargetLanguage);

export const captionerTypes: CaptionOption[] = [
    {
        name: 'AceStepCaptioner',
        label: 'Ace Step',
        group: 'music',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ACE-Step/acestep-transcriber', defaultNameOrPath],
            'config.process[0].caption.model_name_or_path2': ['ACE-Step/acestep-captioner', undefined],
            'config.process[0].caption.extensions': [extensionsAudio, defaultExtensions],
        },
        name_or_path_options: [
            { value: 'ACE-Step/acestep-transcriber', label: 'ACE-Step/acestep-transcriber' },
        ],
        name_or_path2_options: [
            { value: 'ACE-Step/acestep-captioner', label: 'ACE-Step/acestep-captioner' },
        ],
        additionalSections: [
            'caption.model_name_or_path2',
            'caption.fixed_caption',
        ],
        supportsQuantization: true,
        supportsLowVram: true,
    },
    {
        name: 'Qwen3VLCaptioner',
        label: 'Qwen3-VL',
        group: 'image',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.prompt_template': [defaultCaptionPromptTemplate, undefined],
            'config.process[0].caption.target_lang': [defaultCaptionTargetLanguage, undefined],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [128, undefined],

        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
        ],
        additionalSections: [
            'caption.prompt_template',
            'caption.target_lang',
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
        supportsQuantization: true,
        supportsLowVram: true,
    },
    {
        name: 'RemoteAPICaptioner',
        label: '自定义 API',
        group: 'image',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['', defaultNameOrPath],
            'config.process[0].caption.api_base_url': ['', undefined],
            'config.process[0].caption.api_key': ['', undefined],
            'config.process[0].caption.api_protocol': ['openai', undefined],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.prompt_template': [defaultCaptionPromptTemplate, undefined],
            'config.process[0].caption.target_lang': [defaultCaptionTargetLanguage, undefined],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [128, undefined],
            'config.process[0].caption.api_concurrency': [20, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
        },
        additionalSections: [
            'caption.api_base_url',
            'caption.api_key',
            'caption.api_protocol',
            'caption.api_concurrency',
            'caption.prompt_template',
            'caption.target_lang',
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
        supportsQuantization: false,
        supportsLowVram: false,
    },

].sort((a, b) => {
    // Sort by label, case-insensitive
    return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedCaptionerTypes: GroupedSelectOption[] = captionerTypes.reduce((acc, arch) => {
    const groupLabel = arch.group === 'image' ? '图片' : '音频';
    const group = acc.find(g => g.label === groupLabel);
    if (group) {
        group.options.push({ value: arch.name, label: arch.label });
    } else {
        acc.push({
            label: groupLabel,
            options: [{ value: arch.name, label: arch.label }],
        });
    }
    return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
    { value: '', label: '- 不量化 -' },
    { value: 'float8', label: 'float8（默认）' },
    { value: 'uint7', label: '7 bit' },
    { value: 'uint6', label: '6 bit' },
    { value: 'uint5', label: '5 bit' },
    { value: 'uint4', label: '4 bit' },
    { value: 'uint3', label: '3 bit' },
    { value: 'uint2', label: '2 bit' },
];

export const maxResOptions: SelectOption[] = [
    { value: '256', label: '256' },
    { value: '512', label: '512（默认）' },
    { value: '768', label: '768' },
    { value: '1024', label: '1024' },
];
export const maxNewTokensOptions: SelectOption[] = [
    { value: '64', label: '64' },
    { value: '128', label: '128（默认）' },
    { value: '256', label: '256' },
    { value: '512', label: '512' },
    { value: '1024', label: '1024' },
];

export const defaultQtype = 'float8';
