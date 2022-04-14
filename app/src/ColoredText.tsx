import { equals } from "./common";
import { ViewComponentProps } from "./proto";
import {toColor, tokenToParagraphString} from "./ui_common"
export function ColoredTextControlled(props: ViewComponentProps) {
    let { vnt, highlight, setHover, setFocus } = props;
    if (vnt.tensor.shape.length !== 1) {
        let err = `ColoredText only takes 1 dimension, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }
    return (<span style={{whiteSpace:"pre-wrap"}}>
        {vnt.dim_idx_names[0].map((word: string, i: number) => {
            let pick = [i];
            let value = vnt.tensor.get(i);
            let highlightStyle = equals(highlight, pick) ? { outline: "1px solid red", outlineOffset: "-1px" } : {};
            // Previously had different colors depending on if it was hovered, but I don't think we need that.
            // const color = colorSaturation(hover && i === hover[0] ? (hue + 0.5) % 1 : hue, value);
            const handlers = {
                onClick: (e: any) => setFocus(pick, e.target, value),
                onMouseEnter: (e: any) => setHover(pick, e.target, value)
            };
            return (
                <span key={i} style={{
                    backgroundColor: toColor(value,vnt.colorScale),
                    fontSize: "18px",
                    cursor: "pointer",
                    ...highlightStyle
                }} {...handlers}>{tokenToParagraphString( word)}</span>
            );
        })}</span>);
}
