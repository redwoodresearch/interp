import { range, toPercentString } from "./common";
import { PickArgs, VeryNamedTensor, ViewComponentProps } from "./proto";
import { colorSaturation, split_color_wheel, tokenToParagraphString } from "./ui_common";

export function MultiHueText(props: ViewComponentProps) {
    let { setHover, setFocus, highlight,vnt} = props;
    
    const wordsPerRow = 24;
    const maxSaturation = 0.5;
    const numWords = vnt.dim_idx_names[1].length;
    const nRows = Math.ceil(numWords / wordsPerRow);
    const numColors = vnt.dim_idx_names[0].length;
    const hues = split_color_wheel(numColors);
    const heightPercentStr = toPercentString(1 / numColors);
    return (<div style={{ display: "flex", flexDirection: "column", margin: "6px" }}>
        <div>
            {vnt.dim_idx_names[0].map((colorName, colorIdx) => {
                return (<span key={colorIdx} style={{ padding: "3px", backgroundColor: colorSaturation(hues[colorIdx], maxSaturation) }}>{colorName}</span>);
            })}
        </div>
        {range(nRows).map((row) => {
            return (
                <div key={row} style={{ display: "flex", flexDirection: "row", fontSize: "20px" }}>
                    {vnt.dim_idx_names[1].slice(row * wordsPerRow, (row + 1) * wordsPerRow).map((word: string, iRow: number) => {
                        const i = iRow + row * wordsPerRow;
                        return (
                            <div key={i} style={{ position: "relative", top: "1px" }}>{(vnt.dim_idx_names[0]).map((word, j) => {
                                let value = vnt.tensor.get(j, i);
                                const color = colorSaturation(hues[j], 1);
                                return (<div key={j} style={{ pointerEvents: "none", position: "absolute", backgroundColor: color, opacity: Math.min(Math.max(0, value * maxSaturation), 1), height: "100%", width: "100%", }} onMouseEnter={()=>setHover([null, i])} onClick={()=>setFocus([null,i])}></div>);
                            })}<span style={{ pointerEvents: "none", position: "absolute", paddingLeft: word[0] === ' ' ? "4px" : "0", }}>{tokenToParagraphString( word)}</span><span style={{ paddingLeft: word[0] === ' ' ? "4px" : "0", color: "#00000000" }} onMouseEnter={()=>setHover([null, i])} onClick={()=>setFocus([null,i])}>{tokenToParagraphString( word)}</span></div>
                        );
                    })}</div>
            );
        })}
    </div>);
}
