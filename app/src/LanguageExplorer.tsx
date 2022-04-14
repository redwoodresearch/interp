// this takes in a [seq,vocab] VNT where vocab is some subset of the vocabulary that 

import { useState } from "react";
import { range, zeroOneToSaturation } from "./common";
import { ViewComponentProps, VeryNamedTensor, WordPredictionPercent } from "./proto";
import { PercentagesTable } from "./Tables";
import { colorSaturation } from "./ui_common";

// should include all top options for each token
export function LanguageExplorer(props: ViewComponentProps) {
    const topk = 15;
    const [hoveredIdx, setHoveredIdx] = useState(null as any);
    // const [probStyle, setProbStyle] = useState(null as any);
    const vnt: VeryNamedTensor = props.vnt;
    const sorted_idxs = vnt.dim_idx_names[0].map((name, i) => {
        let list = range(vnt.tensor.shape[1]);
        list.sort((a, b) => vnt.tensor.get(i, b) - vnt.tensor.get(i, a));
        list = list.slice(0, topk);
        return list;
    });
    const correct_idxs = vnt.dim_idx_names[0].map((token) => vnt.dim_idx_names[1].indexOf(token));

    let pcts: Array<WordPredictionPercent> = [];
    if (hoveredIdx !== null) {
        pcts = sorted_idxs[hoveredIdx].map((vidx: number) => {
            let word = vnt.dim_idx_names[1][vidx];
            let pct = vnt.tensor.get(hoveredIdx, vidx);
            return { word, pct };
        });
    }

    return (<div>
        <span onMouseLeave={() => setHoveredIdx(null)}>
            {vnt.dim_idx_names[0].map((word: string, i: number) => {
                const value = vnt.tensor.get(i, correct_idxs[i]);
                const color = colorSaturation(0, zeroOneToSaturation(value / 100));
                return (<span key={i} style={{ backgroundColor: color, fontSize: "18px", border: "1px solid black", borderWidth: (i === hoveredIdx ? 1 : 0) }} onMouseEnter={() => {
                    setHoveredIdx(i);
                }}>{word}</span>);
            })}</span>
        <div style={{ minHeight: "500px" }}>
            <PercentagesTable pcts={pcts} ></PercentagesTable>
        </div>
    </div >);
}