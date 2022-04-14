import chroma from "chroma-js";
import { inf, sup } from "ndarray-ops";
import { useState, useRef } from "react";
import { isPickArgsSubset, meanRest, range, toPercentString } from "./common";
import { ViewComponentProps } from "./proto";
import { colorNegPos } from "./ui_common";

export function AttentionLinesControlled(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;
    const svgRef = useRef(null);
    if (vnt.tensor.shape.length !== 2) {
        let err = `AttentionLines only takes 2 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }

    const c = {
        wordHSpacing: 20,
        wordHeight: 16,
        fontSize: 12,
        seperation: 80,
        wordMaxWidth: 80,
        backgroundValueMultiplier: 0.15,
        maxNumTokens: 512,
    };
    const extent = [inf(vnt.tensor), sup(vnt.tensor)];
    function normValue(value: number) {
        return (value - extent[0]) / (extent[1] - extent[0]);
    }
    // Scale from transparent to semi-transparent in [0,1] space

    const wordHOffset = (c.wordHSpacing - c.fontSize) / 2;
    const rectHOffset = (c.wordHSpacing - c.wordHeight) / 2 - c.wordHSpacing;
    const wordStart = 2 * c.wordHSpacing;
    const col2Start = c.wordMaxWidth + c.seperation;

    const numTokens = Math.min(Math.max(vnt.dim_idx_names[0].length, vnt.dim_idx_names[0].length), c.maxNumTokens);
    const neededHeight = (numTokens + 1) * c.wordHSpacing;
    const neededWidth = c.wordMaxWidth * 2 + c.seperation;

    // TODO: to / from direction confusion.

    function displayToken(leftCol: boolean, idx: number, word: string) {
        const pick = leftCol ? [idx, null] : [null, idx];
        const xStart = leftCol ? 0 : col2Start;
        const yStart = wordStart + idx * c.wordHSpacing;

        let value = 0;
        // If other col hovered, change value to corresponding
        if (leftCol && highlight && highlight[1] !== null) {
            value = vnt.tensor.get(idx, highlight[1]);
        } else if (!leftCol && highlight && highlight[0] !== null) {
            value = vnt.tensor.get(highlight[0], idx);
        } else {
            // otherwise, value is mean of others
            value = meanRest(vnt.tensor, pick);
        }
        value = normValue(value);
        const handlers = {
            onClick: (e: any) => setFocus(pick, e.target, value),
            onMouseEnter: (e: any) => setHover(pick, e.target, value)
        };
        return (
            <g transform={`translate(${xStart},${yStart})`} {...handlers}>
                <rect
                    width={c.wordMaxWidth}
                    height={c.wordHeight}
                    transform={`translate(${0},${rectHOffset})`}
                    fill={colorNegPos(value)}
                >
                </rect>
                <text
                    textAnchor={leftCol ? "end" : "begin"}
                    transform={`translate(${leftCol ? c.wordMaxWidth : 0}, ${-wordHOffset})`}>
                    {word}
                </text>
            </g >
        );
    }

    return (<svg ref={svgRef} width={neededWidth} height={neededHeight}>
        <text textAnchor="end" transform={`translate(${c.wordMaxWidth},${c.wordHeight})`}>{vnt.dim_names[0]}</text>
        <text transform={`translate(${c.seperation + c.wordMaxWidth},${c.wordHeight})`}>{vnt.dim_names[1]}</text>
        <g>
            {vnt.dim_idx_names[0].slice(0, numTokens).map((w, i) => displayToken(true, i, w))}
        </g>
        <g>
            {vnt.dim_idx_names[1].slice(0, numTokens).map((w, i) => displayToken(false, i, w))}
        </g>

        <g>
            {range(numTokens).map((to) => (<>
                {range(numTokens).map((from) =>
                    (() => {
                        let value = normValue(vnt.tensor.get(to, from));
                        const handlers = {
                            onClick: (e: any) => setFocus([to, from], e.target, vnt.tensor.get(to, from)),
                            onMouseEnter: (e: any) => setHover([to, from], e.target, vnt.tensor.get(to, from))
                        };
                        if (!isNaN(value) && isPickArgsSubset([to,from],highlight)) {
                            return (
                                <line
                                    x1={c.wordMaxWidth}
                                    x2={col2Start}
                                    y1={wordStart + (to - 0.5) * c.wordHSpacing}
                                    y2={wordStart + (from - 0.5) * c.wordHSpacing}
                                    stroke={`rgba(255,0,0,${toPercentString(value)}`}
                                    strokeWidth={2}
                                    {...handlers} >
                                </line>
                            );
                        }
                    })()
                )}
            </>))}
        </g>

    </svg >);
}
