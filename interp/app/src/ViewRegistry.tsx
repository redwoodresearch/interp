import { AttentionLinesControlled } from "./AttentionLines";
import { ColoredTextControlled } from "./ColoredText";
import { getAxisDimTypes, range, vntPermute, vntPick } from "./common";
import { LanguageExplorer } from "./LanguageExplorer";
import LinePlot from "./LinePlot";
import Histogram from "./Histogram";
import { LazyVeryNamedTensor, PickArgs, ViewComponentProps, viewRegistryEntry, ViewSpec } from "./proto";
import { Scalar } from "./Scalar";
import { SidedMatrix } from "./SidedMatrix";
import { TinyMatrix } from "./ColoredTinyMatrix";
import { MultiHueTinyMatrix } from "./MultiHueTinyMatrix";
import { TextAverageTo } from "./TextAverageTo";
import { Tops } from "./Tops";
import { MultiHueText } from "./MultiHueText";
import { MiniHistogram } from "./MiniHistogram";
import { Barcode } from "./Barcode";
import Unembed from "./Unembed";

const dimPermutedView = (view:(x:ViewComponentProps)=>JSX.Element,permer:(n:number)=>number[])=>{
    return (props:ViewComponentProps)=>{
        const perm = permer(props.highlight.length)
        return view({...props,highlight:perm.map(i=>props.highlight[i]),setFocus:(axes:PickArgs,...args:any[])=>props.setFocus(perm.map(i=>axes[i]),...args),setHover:(axes:PickArgs,...args:any[])=>props.setFocus(perm.map(i=>axes[i]),...args),vnt:vntPermute(props.vnt,perm)})
    }
}

export class ViewRegistry {
    static viewRegistry: viewRegistryEntry[] = [
        { name: 'Stacked tiny matrices', widget: MultiHueTinyMatrix, dimTypes: [], free_dims: 3, min_free_dims: 3 },
        { name: 'Tiny matrix', widget: TinyMatrix, dimTypes: [], free_dims: 2, min_free_dims: 2 },
        { name: 'Barcode', widget: Barcode, dimTypes: [], free_dims: 1, min_free_dims: 1 },
        { name: 'Sided matrix', widget: SidedMatrix, dimTypes: [], free_dims: 2, min_free_dims: 2 },
        { name: 'TextAverageTo', widget: TextAverageTo, dimTypes: ["seq", "seq"], free_dims: 0, min_free_dims: 0 },
        { name: 'TextAverageFrom', widget: dimPermutedView( TextAverageTo, (x)=>range(x).map(i=>x-i-1)), dimTypes: ["seq", "seq"], free_dims: 0, min_free_dims: 0 },
        { name: 'Colored text', widget: ColoredTextControlled, dimTypes: ["seq"], free_dims: 0, min_free_dims: 0 },
        { name: 'Stacked colored Text', widget: MultiHueText, dimTypes: ["seq"], free_dims: 1, min_free_dims: 1 },
        // { name: 'MatAndText', widget: MatAndText, dimTypes: ["seq", "seq"], free_dims: 0 },

        // CM: disabling LanguageExplorer for now - it only works with tensors set up to do topK vocab
        // which we don't currently have in the interface so it just shows garbage
        // { name: 'LanguageExplorer', widget: LanguageExplorer, dimTypes: ['seq', 'vocab'], free_dims: 0, min_free_dims: 0 },

        { name: 'Line plot', widget: LinePlot, dimTypes: [], free_dims: 2, min_free_dims: 1 },

        // CM: Histogram is pretty slow currently, can we speed this up?
        // { name: 'Histogram', widget: Histogram, dimTypes: [], free_dims: 6, min_free_dims: 1 },


        { name: 'Attention lines', widget: AttentionLinesControlled, dimTypes: ["seq", "seq"], dimNames: ["to", "from"], free_dims: 0, min_free_dims: 0 },
        { name: 'Top-k', widget: Tops, dimTypes: ["vocab"], free_dims: 0, min_free_dims: 0 },
        { name: 'Scalar', widget: Scalar, dimTypes: [], free_dims: 0, min_free_dims: 0 },
        { name: 'MiniHistogram', widget: MiniHistogram, dimTypes: [], free_dims: 10, min_free_dims: 1 },
        { name: 'Unembed', widget: Unembed, dimTypes: ["hidden"], free_dims: 0, min_free_dims: 0 },
    ];

    static getEntry(name: string) {
        let vre = ViewRegistry.viewRegistry.find(e => e.name === name);
        if (!vre) {
            throw new Error('Failed to find entry with name: ' + name);
        }
        return vre;
    }

    static getVizes(axisDimTypes: string[]) {
        let viableVizes = ViewRegistry.viewRegistry.filter(vr =>
            (axisDimTypes.length <= vr.dimTypes.length + vr.free_dims) &&
            (axisDimTypes.length >= vr.dimTypes.length + vr.min_free_dims)
        );
        const narrowerVizes = ViewRegistry.viewRegistry.filter(vr => {
            let axisDimTypesTemp = [...axisDimTypes];
            for (let axisType of vr.dimTypes) {
                if (!axisDimTypesTemp.includes(axisType)) {
                    return false;
                } else {
                    axisDimTypesTemp.splice(axisDimTypesTemp.indexOf(axisType), 1);
                }
            }
            return true;
        });
        narrowerVizes.sort((a, b) => b.dimTypes.length - a.dimTypes.length);
        const typeMatchingVizes = ViewRegistry.viewRegistry.filter(vr =>
            narrowerVizes.indexOf(vr) !== -1 && viableVizes.indexOf(vr) !== -1
        );
        typeMatchingVizes.sort((a, b) => b.dimTypes.length - a.dimTypes.length);
        viableVizes = [...typeMatchingVizes, ...viableVizes.filter(x => typeMatchingVizes.indexOf(x) === -1)];
        narrowerVizes.sort((a, b) => b.dimTypes.length + b.min_free_dims - (a.dimTypes.length + a.min_free_dims));
        return { viableVizes, typeMatchingVizes, narrowerVizes };
    }

    static getVizOrSubstitute(lvnt: LazyVeryNamedTensor, spec: ViewSpec, requestedVizName?: string) {
        const axisDimTypes = getAxisDimTypes(lvnt.dim_types, spec);
        const { viableVizes } = ViewRegistry.getVizes(axisDimTypes);

        if (requestedVizName && viableVizes.find(viz => viz.name === requestedVizName)) {
            return requestedVizName;
        } else if (viableVizes.length > 0) {
            return viableVizes[0].name;
        } else {
            return "Scalar";
        }
    }


}
