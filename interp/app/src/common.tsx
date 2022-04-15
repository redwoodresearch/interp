import ndarray from "ndarray";
import { inf, sup } from "ndarray-ops";
import { AttribLocation, AttributionStateSpec, AttributionTensors, AttributionTreeTensors, LazyVeryNamedTensor, PickArgs, SparseTensor, VeryNamedTensor, ViewSpec } from "./proto";
import { useLocation } from "react-router-dom";
import React from "react";

export function useQuery() {
    const { search } = useLocation();
    return React.useMemo(() => new URLSearchParams(search), [search]);
}

export const focusIsEmpty = (focus: PickArgs) => {
    return focus.every(x => x === null);
};
export const getClearedFocus = (focus: PickArgs) => {
    return focus.map(x => null);
};

export const focusEquals = (a: PickArgs, b: PickArgs) => {
    return (a.length === b.length) && a.every((x, i) => x === b[i]);
};


// mutates the input :( but whatevs
export const deepPromiseAll: (x: any) => Promise<any> = (x: any) => {
    const promiseList = [] as Promise<any>[];

    const recurse = (x: any) => {
        if (typeof x === "object") {
            if (x instanceof Promise) {
                promiseList.push(x);
            } else if (Array.isArray(x)) {
                x.forEach(recurse);
            } else {
                for (let key in x) {
                    recurse(x[key]);
                }
            }
        }
    };

    recurse(x);

    const listPromise = Promise.all(promiseList);
    return listPromise.then((results: any[]) => {
        let i = 0;
        const recurse = (x: any) => {
            if (typeof x === "object") {
                if (x instanceof Promise) {
                    i++;
                    return results[i - 1];
                } else if (Array.isArray(x)) {
                    for (let i = 0; i < x.length; i++) {
                        x[i] = recurse(x[i]);
                    }
                } else {
                    for (let key in x) {
                        x[key] = recurse(x[key]);
                    }
                }
            }
            return x;
        };
        return recurse(x);
    });
};

export const bisect = (arr: any[], val: any) => {
    let top = arr.length, bottom = 0;
    for (let i = 0; i < 100; i++) {
        const middle = Math.floor((top + bottom) / 2);

        if (arr[middle] < val) {
            bottom = middle;
        } else {
            top = middle;
        }
    }
    return top;
};

export const range = (x: number) => {
    const result = [];
    for (let i = 0; i < x; i++) {
        result.push(i);
    }
    return result;
};

export const meanRest = (ndarr: ndarray.NdArray, picks: PickArgs, ignoreNaN: boolean = true) => {
    let result = 0;
    const picked = ndarr.pick(...picks);
    let numElements = ndarr.shape.reduce((a, b) => a * b, 1) / picked.shape.reduce((a: number, b) => a * b, 1);
    const fn = (arr: ndarray.NdArray) => {
        if (arr.shape.length > 1) {
            for (let i = 0; i < arr.shape[0]; i++) {
                fn(arr.pick(i));
            }
        } else {
            for (let i = 0; i < arr.shape[0]; i++) {
                const el = arr.get(i);
                if (ignoreNaN && isNaN(el)) {
                    numElements -= 1;
                } else {
                    result += arr.get(i);
                }
            }
        }
    };
    fn(picked);
    if (numElements > 0) {
        result /= numElements;
        return result;
    } else {
        return NaN;
    }
};

export const toPercentString = (x: number) => {
    return `${Math.floor(x * 100)}%`;
};

export function zeroOneToSaturation(value: number) {
    // Custom log-ish scale
    if (value >= 1e-1) return 0;
    if (value >= 1e-2) return 0.2;
    if (value >= 1e-3) return 0.4;
    if (value >= 1e-4) return 0.6;
    if (value >= 1e-5) return 0.8;
    return 1;
}



export const maskToInverseIndex = (mask: boolean[]) => {
    let iidx = 0;
    const result = mask.map(() => -1);
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) {
            result[i] = iidx;
            iidx += 1;
        }
    }
    return result;
};

// passed function can't modify idx! I need that!
export const ndarrayWalk = (ndarr: ndarray.NdArray, func: any) => {
    const idx = ndarr.shape.map(x=>0)
    let curDim = idx.length-1
    while(true){
        curDim=idx.length-1
        func(idx,ndarr.get(...idx))
        idx[curDim] = (idx[curDim]+1)%ndarr.shape[curDim]
        while(idx[curDim]===0){
            curDim-=1
            idx[curDim]=(idx[curDim]+1)%ndarr.shape[curDim]
        }
        if (curDim===-1){
            break
        }
    }
};

export const getOverX = (ndarr: ndarray.NdArray, x: number, doNegative: boolean = true) => {
    const result = { idxs: [], values: [] } as SparseTensor;
    const fn = (arr: ndarray.NdArray, idx: number[]) => {
        if (arr.shape.length > 1) {
            for (let i = 0; i < arr.shape[0]; i++) {
                fn(arr.pick(i), [...idx, i]);
            }
        } else {
            for (let i = 0; i < arr.shape[0]; i++) {
                const value = arr.get(i);
                const abs_value = doNegative ? Math.abs(value) : value;
                if (abs_value > x) {
                    result.idxs.push([...idx,i]);
                    result.values.push(value);
                }
            }
        }
    };
    fn(ndarr, []);
    return result;
};

export const vntPick = (vnt: VeryNamedTensor, myArgs: PickArgs) => {
    if (vnt.tensor.shape.length !== vnt.dim_names.length || vnt.tensor.shape.length !== vnt.dim_types.length || vnt.tensor.shape.length !== vnt.dim_idx_names.length) {
        console.log(vnt);
        throw new Error("pick input inconsistent dims above");
    }
    if (vnt.tensor.shape.length !== myArgs.length) {
        throw new Error(`${vnt.tensor.shape} can't pick ${myArgs}`);
    }
    return {
        title: myArgs.map((x, i) => x !== null ? `${vnt.dim_names[i]} ${vnt.dim_idx_names[i][x]}` : '').join(", "),
        tensor: vnt.tensor.pick(...myArgs),
        dim_names: vnt.dim_names.filter((x, i) => myArgs[i] === null || myArgs[i] === -1),
        dim_types: vnt.dim_types.filter((x, i) => myArgs[i] === null || myArgs[i] === -1),
        dim_idx_names: vnt.dim_idx_names.filter((x, i) => myArgs[i] === null || myArgs[i] === -1),
        units: vnt.units,
        colorScale: vnt.colorScale
    } as VeryNamedTensor;
};

export const sumNdarrays = (...ndarrs: ndarray.NdArray[]) => {
    const result = ndarray(new Float32Array(ndarrs[0].shape.reduce((a, b) => a * b, 1)), ndarrs[0].shape);
    const fn = (arr: ndarray.NdArray, result: ndarray.NdArray) => {
        if (arr.shape.length > 1) {
            for (let i = 0; i < arr.shape[0]; i++) {
                fn(arr.pick(i), result.pick(i));
            }
        } else {
            for (let i = 0; i < arr.shape[0]; i++) {
                result.set(i, arr.get(i) + result.get(i));
            }
        }
    };
    ndarrs.forEach((x) => fn(x, result));
    return result;
};


export const areAxesDifferent = (a: ViewSpec, b: ViewSpec) => {
    return a.some((newAx, i) => (newAx === 'axis' && b[i] !== 'axis') || (newAx !== 'axis' && b[i] === 'axis'));
};

const constrainedLVNT = (lvnt: LazyVeryNamedTensor, outerViewSpec: ViewSpec) => {
    const mask = outerViewSpec.map(x => x === "axis");
    const result = {
        _getView: (viewSpec: ViewSpec) => {
            const vs = matchArrToHoles(mask, outerViewSpec, viewSpec);
            const result = lvnt._getView(vs);
            return result;
        },
        _getSparseView: (picks: PickArgs[], threshold: number) => {
            const vs = picks.map((pick, i) => matchArrToHoles(mask, outerViewSpec, pick));
            if (vs.some((x: any) => x.some((y: any) => y instanceof String))) {
                console.log(vs, outerViewSpec, picks);
                throw new Error("can only get sparse view of indexes, not reductions");
            }
            const result = lvnt._getSparseView(vs as any, threshold);
            return result.then((sparseTensor) => {

                return { idxs: sparseTensor.idxs.map(x => x.filter((a, i) => mask[i])), values: sparseTensor.values } as SparseTensor;
            });
        },
        dim_names: lvnt.dim_names.filter((x, i) => outerViewSpec[i] == "axis"),
        dim_types: lvnt.dim_types.filter((x, i) => outerViewSpec[i] == "axis"),
        dim_idx_names: lvnt.dim_idx_names.filter((x, i) => outerViewSpec[i] == "axis"),
    } as unknown as LazyVeryNamedTensor;
    return result;
};

export const vntPermute = (vnt: VeryNamedTensor, permutation: number[]) => {
    const result = {
        title: vnt.title,
        tensor: vnt.tensor.transpose(...permutation),
        dim_names: permutation.map(x => vnt.dim_names[x]),
        dim_types: permutation.map(x => vnt.dim_types[x]),
        dim_idx_names: permutation.map(x => vnt.dim_idx_names[x]),
        units: vnt.units,
        colorScale: vnt.colorScale,
    } as VeryNamedTensor;
    return result;
};
export const lvntPermute = (lvnt: LazyVeryNamedTensor, permutation: number[]) => {
    const inversePermutation = range(permutation.length);
    for (let i = 0; i < permutation.length; i++) {
        inversePermutation[permutation[i]] = i;
    }
    const result = {
        _getView: (viewSpec: ViewSpec) => {
            const vs = inversePermutation.map((i) => viewSpec[i]);
            const result = lvnt._getView(vs);
            if (result instanceof Promise) {
                return result.then(vnt => {
                    return vntPermute(vnt, inversePermutation);
                });
            }
            return vntPermute(result, inversePermutation);
        },
        _getSparseView: (picks: PickArgs[], threshold: number) => {
            const vs = picks.map((pick) => inversePermutation.map((i) => pick[i]));
            const result = lvnt._getSparseView(vs, threshold);
            return result.then((sparseTensor) => {
                return { idxs: sparseTensor.idxs.map(x => inversePermutation.map(pidx => x[pidx])), values: sparseTensor.values } as SparseTensor;
            });
        },
        dim_names: permutation.map(x => lvnt.dim_names[x]),
        dim_types: permutation.map(x => lvnt.dim_types[x]),
        dim_idx_names: permutation.map(x => lvnt.dim_idx_names[x]),
        units: lvnt.units,
    } as LazyVeryNamedTensor;
    return result;
};

// @TODO: make this index from an existing cached view if that's possible
// export const lvntGet:
//     ((lvnt: LazyVeryNamedTensor, viewSpec: (string | number | null)[]) => VeryNamedTensor | Promise<VeryNamedTensor>)

//     = (lvnt: LazyVeryNamedTensor, viewSpec: (string | number | null)[]) => {
//         const viewKey = JSON.stringify(viewSpec);
//         if (lvnt._viewCache === undefined) {
//             lvnt._viewCache = {};
//         }
//         if (lvnt._viewCache[viewKey] !== undefined) {
//             return lvnt._viewCache[viewKey];
//         }
//         const promise = lvnt._getView(viewSpec);
//         promise.then(x => { lvnt._viewCache[viewKey] = x; });
//         return promise;
//     };


export const matchDimOrder = (vnt: VeryNamedTensor | LazyVeryNamedTensor, order: string[]) => {
    let newTypes = [...vnt.dim_types];
    let permutation = range(vnt.dim_types.length);
    const lenDiff = newTypes.length - order.length;
    for (let i = order.length - 1; i >= 0; i--) {
        if (order[i] !== newTypes[i + lenDiff]) {
            const idx = newTypes.indexOf(order[i]);

            const tmpperm = permutation[idx];
            permutation[idx] = permutation[i];
            permutation[i] = tmpperm;

            const tmpnt = newTypes[idx];
            newTypes[idx] = newTypes[i];
            newTypes[i] = tmpnt;
        }
    }
    if ((vnt as VeryNamedTensor).tensor) {
        vnt = vnt as VeryNamedTensor;
        const result = vntPermute(vnt, permutation);
        return result;
    } else {
        vnt = vnt as LazyVeryNamedTensor;
        const result = lvntPermute(vnt, permutation);
        return result;
    }
};

export function matchArrToHoles<Type>(mask: boolean[], outer: Type[], inner: Type[]) {
    // write values from inner, preserving their order, into positions where mask[outer] is True
    // TODO: should assert that all values of inner are written
    let maskLen = mask.filter(x => x).length;
    if (maskLen != inner.length) {
        console.error(`Logic error: have ${maskLen} holes but only ${inner.length} values to fill!`);
    }

    let outerCopy = [...outer];
    let innerIdx = 0;
    for (let outerIdx = 0; outerIdx < outerCopy.length; outerIdx++) {
        if (mask[outerIdx]) {
            outerCopy[outerIdx] = inner[innerIdx];
            innerIdx += 1;
        }
    }
    return outerCopy;
};




export function getAxisDimTypes(dim_types: string[], spec: ViewSpec) {
    if (dim_types.length !== spec.length) {
        throw new Error(`Expected dim types and spec to have same length! types: ${dim_types} spec; ${spec}`);
    }
    return dim_types.filter((x, i) => spec[i] === "axis");
}
export const lvntGet:
    ((lvnt: LazyVeryNamedTensor, viewSpec: (string | number | null)[]) => VeryNamedTensor | Promise<VeryNamedTensor>)

    = (lvnt: LazyVeryNamedTensor, viewSpec: (string | number | null)[]) => {
        const viewKey = JSON.stringify(viewSpec);
        if (lvnt._viewCache === undefined) {
            lvnt._viewCache = {};
        }
        if (lvnt._viewCache[viewKey] !== undefined) {
            return lvnt._viewCache[viewKey];
        }
        let promise = lvnt._getView(viewSpec);
        promise = promise.then(x => { x.colorScale = Math.max(-inf(x.tensor), sup(x.tensor)); lvnt._viewCache[viewKey] = x; return x; });
        return promise;
    };

export function getHighlight(focus: PickArgs, hover: PickArgs): PickArgs {
    if (hover.find(h => h !== null)) {
        return hover;
    }
    return focus;
}

export function equals(highlight: PickArgs, pick: PickArgs) {
    return (highlight.length === pick.length) && highlight.every((x, i) => x === pick[i]);
}

export const deepEquals: (a: any, b: any) => any = (a: any, b: any) => {
    if (typeof a !== typeof b) return false;
    if (typeof a === "object") {
        if (Array.isArray(a) !== Array.isArray(b)) return false;
        if (Array.isArray(a)) {
            return a.length === b.length && a.map((ai, i) => deepEquals(ai, b[i])).every(x => x);
        } else {
            const usedKeys = {};
            for (let key in a) {
                if (!deepEquals(a[key], b[key])) return false;
            }
            for (let key in b) {
                if (!deepEquals(a[key], b[key])) return false;
            }
            return true;
        }

    }
    return a === b;
};

export const isOneArrayPrefixOfOther = (a: any[], b: any[]) => {
    for (let i = 0; ; i++) {
        if (a.length <= i || b.length <= i) {
            return true;
        }
        if (!deepEquals(a[i], b[i])) {
            return false;
        }
    }
};

export const stateToFakeTree = (state: AttributionTensors | AttributionStateSpec, layerNames: string[]) => {
    return { attribution: state.root.attribution, children: state.tree, idx: { token: state.root.data.seqIdx, layerWithIO: layerNames.length - 1, headOrNeuron: 0, isMlp: false } as AttribLocation, threshold: state.root.threshold, outgoingLines: state.root.outgoingLines } as AttributionTreeTensors;
};

export const isDeepSubtree = (tree: any, subtree: any) => {
    if (typeof subtree !== typeof tree) {
        return false;
    }
    if (typeof subtree === "object") {
        if (Array.isArray(subtree)) {
            throw Error("is deep subtree not implemented for arrays");
        } else {
            for (let subtreeKey in subtree) {
                const subtreeVal = subtree[subtreeKey];
                const treeVal = tree[subtreeKey];
                if (!isDeepSubtree(treeVal, subtreeVal)) {
                    return false;
                }
            }
            return true;
        }
    } else {
        return tree === subtree;
    }
};

export const isPickArgsSubset = (inner:PickArgs,outer:PickArgs)=>{
    return inner.every((ai,i)=>outer[i]===null||inner[i]===outer[i])
}