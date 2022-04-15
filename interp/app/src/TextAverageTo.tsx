import ndarray from "ndarray";
import { ColoredTextControlled } from "./ColoredText";
import { meanRest, vntPick } from "./common";
import { ViewComponentProps, PickArgs, SetHoverCallback, SetFocusCallback } from "./proto";

export function TextAverageTo(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;
    let vntInner;
    if (highlight && highlight[0] !== null) {
        vntInner = vntPick(vnt, [highlight[0], null]);
    } else {
        const result = vntPick(vnt, [0, null]);
        const sums = [];
        const counts = [];
        const means = [];
        // for row in arr
        for (let i = 0; i < vnt.tensor.shape[1]; i++) {
            sums.push(0);
            counts.push(0);
            for (let j = 0; j < vnt.tensor.shape[0]; j++) {
                let val = vnt.tensor.get(j, i);
                if (!isNaN(val)) {
                    sums[i] += val;
                    counts[i] += 1;
                }
            }
            means[i] = (counts[i] > 0) ? sums[i] / counts[i] : NaN;
        }
        result.tensor = ndarray(means, [vnt.tensor.shape[1]]);
        vntInner = result;
    }
    const highlightInner = highlight && highlight.slice(0, 1);

    // We are a 2D widget but our child is only 1D
    let setHoverInner: SetHoverCallback = (axes, el, value) => {
        if (axes.length !== 1) throw new Error();
        setHover([axes[0], null], el, value);
    };
    let setFocusInner: SetFocusCallback = (axes, el, value) => {
        if (axes.length !== 1) throw new Error();
        setFocus([axes[0], null], el, value);
    };

    return (<ColoredTextControlled setHover={setHoverInner} setFocus={setFocusInner} highlight={highlightInner} vnt={vntInner} />);
};
