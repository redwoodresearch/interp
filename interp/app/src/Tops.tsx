import { range,  equals } from "./common";
import { ViewComponentProps } from "./proto";
import { toColor, tokenToInlineString } from "./ui_common";


export function Tops(props: ViewComponentProps) {
    const { vnt, setFocus, setHover, highlight } = props;

    if (vnt.tensor.shape.length !== 1) {
        let err = `Tops only takes 1 dimension, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }

    const topk = 22;
    let sortedIdxs = range(vnt.tensor.shape[0]);
    sortedIdxs.sort((a, b) => vnt.tensor.get(b) - vnt.tensor.get(a));
    sortedIdxs = sortedIdxs.slice(0, topk);


    let makeRow = (vidx: number) => {
        let pick = [vidx];
        let style = {
            cursor: "pointer",
            ...(equals(highlight, pick) ? { outline: "1px solid red", outlineOffset: "-1px" } : {})
        };
        const handlers = {
            onClick: (e: any) => setFocus(pick, e.target, vnt.tensor.get(vidx)),
            onMouseEnter: (e: any) => setHover(pick, e.target, vnt.tensor.get(vidx))
        };
        return (
            <tr style={style} {...handlers}>
                <td>{tokenToInlineString(vnt.dim_idx_names[0][vidx])}</td>
                <td style={{ backgroundColor: toColor(vnt.tensor.get(vidx),vnt.colorScale) }}>
                    {` ${vnt.tensor.get(vidx).toFixed(4)}`}
                </td>
            </tr >);
    };


    return (<div>
        <table>
            <thead>
                <tr>
                    <th>{vnt.dim_names[0]}</th>
                    <th>Value ({vnt.units})</th>
                </tr>
            </thead>
            <tbody>
                {sortedIdxs.map(makeRow)}
            </tbody>
        </table>
    </div>);
}
