import * as ndarray from "ndarray";

export type ViewSpecDim = ("axis" | "facet" | string | number | null);
export type ViewSpec = ViewSpecDim[];
export interface Obj { [key: string]: any; }
export type PickArg = (number | null);
export type PickArgs = PickArg[];
export interface SparseTensor {
    idxs: number[][];
    values: number[];
}
export type Edges = { to: AttribLocation, from: AttribLocation, value: number; }[];

export interface ViewComponentOptions {
    [key: string]: any;
    color?: string;
    extent?: [number, number];
}

export interface VeryNamedTensor {
    title: string;
    dim_names: string[];
    dim_types: string[];
    dim_idx_names: string[][];
    units: string;
    tensor: ndarray.NdArray;
    colorScale?: number;
}

export interface LazyVeryNamedTensor {
    title: string;
    dim_names: string[];
    dim_types: string[];
    dim_idx_names: string[][];
    units: string;
    _getView: (x: ViewSpec) => Promise<VeryNamedTensor>;
    _getSparseView: (target_idxs: PickArgs[], threshold: number) => Promise<SparseTensor>;
    _viewCache: { [viewSpecStr: string]: VeryNamedTensor; };
}

export interface Attributions {
    embeds: VeryNamedTensor; // seq
    heads: VeryNamedTensor; // layer head seq
    mlps: VeryNamedTensor; // layer neuron seq
    __notmanyout__: boolean;
}
export interface AttributionsManyOut {
    embeds: VeryNamedTensor; // o seq
    heads: VeryNamedTensor; // o layer head seq
    mlps: VeryNamedTensor; // o layer neuron seq
    __manyout__: boolean;
}

export interface AttributionBackend {
    layerNames: string[],
    headNames: string[],
    neuronNames: string[],
    tokens: string[],
    hasMlps: boolean;
    modelName:string;
    _startTree: (root: AttributionRoot, useIgOutput: boolean, subtractDatasetMean: boolean, fuseNeurons: boolean) => Promise<AttributionsManyOut>;
    _expandTreeNode: (target_idxs: AttributionPath, useIGAttn: boolean, fuseNeurons: boolean,halfLinearActivation:boolean) => Promise<AttributionsManyOut>;
    _sparseLogitsForSpecificPath: (path: AttributionPath, fake_log_probs?: string, fake_attn?: string, specificLogits?: string[], fusedNeurons?: boolean) => Promise<QueriedLogits>;
    _searchAttributionsFromStart: (threshold: number, edgeThreshold: number, useNeg: boolean, useIGAttn: boolean) => Promise<AttribSet>;
}

export interface QueriedLogits {
    top: FewLogits;
    specific: FewLogits;
    bottom: FewLogits;

}

export interface AncestorFocus {
    dim_type: string;
    dim_name: string;
    dim_idx_names: string[];
    // An unique identifier for some dimension of an ancestor
    key: string;

    // Current value of the dimension identified by "key"
    pick: ViewSpecDim;

    // The index of the dimension in the ancestor
    i: number;
    vizIdx: number;
}

export type AncestralFocus = AncestorFocus[];

export interface OldViewManagerProps {
    lvnts: LazyVeryNamedTensor[];
    defaultSpec?: ViewSpec;
    options: Obj;
}

export interface ChangeRequest {
    vizName?: string;
    viewSpec?: ViewSpec;
    focus?: PickArgs;
    lvntIdx?: number;
    hover?: PickArgs;
    refreshAllViews?: boolean;
    timestamp?:number;
}

export interface NewViewManagerProps {
    ancestralFocus: AncestralFocus;
    lvnt: LazyVeryNamedTensor;
    view?: VeryNamedTensor;
    hover: PickArgs;
    // For each axis, what we are currently doing with that axis
    spec: ViewSpec,

    // Miscellaneous items - currently only stores colormap extent (?)
    options: Obj;

    // The name of a widget class - see ViewRegistry
    vizName: string;

    // For each axis, null or an integer if a specific indexes is selected
    focus: PickArgs;

    // For each axis: null or an integer if a specific index is hovered

    onChangeSpec: (change: ChangeRequest) => void;
    onDelete: () => void;
    onDuplicate: () => void;
    unembedder?:any;
}



export type SetHoverCallback = (axes: PickArgs, el?: HTMLElement, value?: number) => void;
export type SetFocusCallback = SetHoverCallback;

export interface ViewComponentProps {
    vnt: VeryNamedTensor;
    setHover: SetHoverCallback;
    hover?: PickArgs;
    // hover if hover if active, otherwise focus, otherwise nulls
    highlight: PickArgs;
    setFocus: SetFocusCallback;
    options?: ViewComponentOptions;
    unembedder?:any;
}
export interface ViewManagerProps {
    lvnt: LazyVeryNamedTensor;
    defaultViewSpec: ViewSpec;
    viewSpecForce: boolean[];
    options: Obj;
    unembedder?:any;
}

export interface WordPrediction {
    word: string;
    logProb: number;
}

export interface WordPredictionPercent {
    word: string;
    pct: number;
}

export interface viewRegistryEntry {
    name: string,
    widget: (props: ViewComponentProps) => JSX.Element,
    dimTypes: string[],
    free_dims: number,
    dimNames?: string[],
    min_free_dims: number;
}

export interface AttribLocation {
    token: number;
    layerWithIO: number;
    headOrNeuron: number;
    isMlp: boolean;
    qkv?: number;
    headReplica?: number;
};

export interface LocationUsedAndBase {
    base: AttribLocation;
    used?: AttribLocation;
}

export type AttributionPath = AttribLocation[];
export interface AttributionTreeSpec {
    children: AttributionTreeSpec[];
    idx: AttribLocation;
    threshold: number;
}

export interface AttributionTreeTensors {
    children: AttributionTreeTensors[];
    attribution: Attributions;
    outgoingLines?: LocationUsedAndBase[];
    idx: AttribLocation;
    usedIdx: AttribLocation;
    threshold: number;
}

export interface AttributionStateSpec {
    tree: AttributionTreeSpec[];
    root: AttributionRoot;
    threshold: number;
    useIGAttn: boolean;
    useIGOutput: boolean;
    useActivationVsMean: boolean;
    showNegative: boolean;
    fuseNeurons: boolean;
    halfLinearActivation:boolean;
    lineWidthScale: number;
    specificLogits: string[];
    nonce: number;
    modelName:string;
    toks:string[];
}

export interface AttributionTensors {
    tree: AttributionTreeTensors[];
    pathDirectionLogits: null | QueriedLogits;
    root: AttributionRoot;
    nonce: number;
}

export interface LogitRoot {
    seqIdx: number;
    tokString: string;
    comparisonTokString: string | null;
}
export type LogProbRoot = LogitRoot;
export type ProbRoot = LogitRoot;

export interface AttentionPatternRoot {
    seqIdx: number;
    headIdx: number;
    layerIdx: number;
};
export type AttentionPatternRootFrom = AttentionPatternRoot;

export type AttentionSingleRoot = {
    seqIdx: number;
    fromIdx: number;
    headIdx: number;
    layerIdx: number;
};
const TREE_ROOT_OPTIONS = [{ name: "logprob", layerType: "output", direction: "backward" }, { name: "logit", layerType: "output", direction: "backward" }, { name: "attention_pattern", layerType: "head", direction: "backward" }, { name: "attention_pattern_from", layerType: "head", direction: "backward" }, { name: "attention_single", layerType: "head", direction: "backward" }];
export interface AttributionRoot {
    kind: "logprob" | "logit" | "prob" | "attention_pattern_from" | "attention_pattern" | "attention_single";
    data: AttentionSingleRoot | AttentionPatternRoot | AttentionPatternRootFrom | LogitRoot | LogProbRoot | ProbRoot;
    threshold: number;
    attribution?: Attributions;
    outgoingLines?: LocationUsedAndBase[];
}
export const IMPLEMENTED_TOKEN_ROOTS = {logprob:true,logit:true,prob:true} as {[k:string]:boolean}

export interface FewLogits {
    words: string[];
    values: number[];
}

export interface PanelUrlState {
    spec: ViewSpec,

    // Miscellaneous items - currently only stores colormap extent (?)
    options: Obj;

    // The name of a widget class - see ViewRegistry
    vizName: string;

    // For each axis, null or an integer if a specific indexes is selected
    focus: PickArgs;

    hover: PickArgs;

    lvntIdx: number;
}

export interface PanelBigState {
    view?: VeryNamedTensor,
    usedSpec: ViewSpec,
    usedLvntIdx: number,
}

export interface ComposableUIUrlState {
    lvntDims: number[][];
    panels: PanelUrlState[];
    nonce: number;
}
export interface InterpsiteUrlState {
    prompt: string;
    whichModel: number;
    nonce: number;
    allNonce: number;
    attributionUI?: AttribSetSpec | AttributionStateSpec;
    composableUI?: ComposableUIUrlState;
    whichAttributionUI: "set" | "tree";
}

export interface AttribSet {
    nodeValues: number[];
    locations: AttribLocation[];
    edges: Edges;
}

export interface AttribSetSpec {
    root: AttributionRoot;
    threshold: number;
    useIGAttn: boolean;
    useIGOutput: boolean;
    useActivationVsMean: boolean;
    showNegative: boolean;
    // fuseNeurons:boolean;
    
    lineWidthScale: number;
    specificLogits: string[];
    nonce: number;
}

export interface AttribSetState {
    set: null | AttribSet;
    pathDirectionLogits: null | QueriedLogits;
    nonce: number;
}