import React, { useEffect, useRef, useState } from 'react';
import { bisect } from "./common";
import { QueriedLogits } from './proto';

export const Select = (props: any) => {
    let { sortedWords, value, onChange, defaultTopWords, placeholder, showClearButton } = props;
    if (value === null) value = "";

    const [focused, setFocused] = useState(false);
    const [topWords, setTopWords] = useState(defaultTopWords);
    const [typed, setTyped] = useState("");
    const inputRef = useRef(null as null | HTMLInputElement);
    const handleChange = (typed: string) => {
        console.log({ typed });
        const bottom = bisect(sortedWords, typed);
        console.log("bottom", bottom);
        setTopWords(sortedWords.slice(bottom, bottom + 20));
        setTyped(typed);
    };
    return (<div style={{ margin: "5px 0" }} onFocus={() => setFocused(true)} ><input style={{ fontSize: "inherit" }} ref={inputRef} placeholder={placeholder} onChange={e => handleChange(e.target.value)} onKeyDown={(e) => {
        console.log({ eventkey: e.key });
        if (e.key === "Tab" || e.key === "Enter") {
            onChange(topWords[0]);
            if (!e.shiftKey) {
                inputRef.current?.blur();
            }
        }
    }}
        value={focused ? typed : value}
        onBlur={() => setFocused(false)} />
        {showClearButton && <button className="button" onClick={() => { onChange(null); setFocused(false); }}>Clear</button>} {focused && (<div>{topWords.map((w: any, i: number) => (<div key={`${i} ${w}`} style={{ cursor: "pointer" }} onMouseDown={e => e.preventDefault()} onClick={() => { onChange(w); inputRef.current?.blur(); }}>{tokenToInlineString(w)}</div>))}</div>)}</div >);
};

export function ShowQueriedLogits(props: { ql: QueriedLogits; }) {
    return (<div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", minHeight: "500px" }}>
        {["top", "bottom", "specific"].map((wordListName, i) => (
            <div style={{ margin: "0 5px", display: "flex", flexDirection: "column" }}>
                <span>{wordListName.replace(/^\w/, (c) => c.toUpperCase())}: (logits) </span>
                {
                    (props.ql as any)[wordListName].values.map((val: number, i: number) => {
                        return (<span style={{ width: "100%", backgroundColor: colorNegPos(val / 10 /* HACK /10 just because */) }}>
                            <span style={{ display: "block", float: "left" }}> {tokenToInlineString((props.ql as any)[wordListName].words[i])}</span>
                            <span style={{ display: "block", float: "right" }}>{val.toPrecision(3)}</span>
                        </span>);
                    })}
            </div>))}
    </div>);
}

export function truncateElipsis(string:string,length=20){
    if (string.length>length){
        return string.slice(0,length-1)+"…"
    }
    return string
}

export function tokenToInlineString(token: string) {
    return truncateElipsis(token,20).replace("\n", "↵").replace(" ", "⸱");
}

export function tokenToParagraphString(token: string) {
    return token.replace("\n", "↵\n");
}

export function colorSaturation(hue: number, saturation: number) {
    return `hsl(${Math.floor(hue * 360)}, 100%, ${Math.max(Math.min(Math.floor((1 - saturation * 0.5) * 100), 100), 50)}%)`;
}

export function colorNegPos(value: number, huePos: number = 0, hueNeg: number = 0.6) {
    const hue = value > 0 ? huePos : hueNeg;
    const sat = Math.abs(value);
    return colorSaturation(hue, sat);
}

export function toColor(value: number, scale?: number) {
    if (!scale) scale = 1;
    return colorNegPos(value / scale);
}



export function toColorRGBA(value: number, scale?: number, huePos: number = 0, hueNeg: number = 0.6) {
    // from https://css-tricks.com/converting-color-spaces-in-javascript/

    // Calculates HSL (H on [0, 360), s on [0,1], l on [0,1])
    if (scale) value /= scale;
    let h = (value > 0 ? huePos : hueNeg) * 360;
    let s = 1;
    let l = Math.max(Math.min(1 - Math.abs(value) * 0.5, 1), 0.5);

    // Convert to integer RGB on [0, 255]
    let c = (1 - Math.abs(2 * l - 1)) * s,
        x = c * (1 - Math.abs((h / 60) % 2 - 1)),
        m = l - c / 2,
        r = 0,
        g = 0,
        b = 0;

    if (0 <= h && h < 60) {
        r = c; g = x; b = 0;
    } else if (60 <= h && h < 120) {
        r = x; g = c; b = 0;
    } else if (120 <= h && h < 180) {
        r = 0; g = c; b = x;
    } else if (180 <= h && h < 240) {
        r = 0; g = x; b = c;
    } else if (240 <= h && h < 300) {
        r = x; g = 0; b = c;
    } else if (300 <= h && h < 360) {
        r = c; g = 0; b = x;
    }

    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);

    return [r, g, b, 255];
}

export function css_prop_by_id(id: any, prop: any) {
    let el = document.getElementById(id);
    if (el) return window.getComputedStyle(el).getPropertyValue(prop);
    return "";
}




export function NumTooltip(props: { value: number, units: string, targetRef: HTMLElement; }) {
    const { targetRef, units, value } = props;

    const [nonce, setNonce] = useState(0);
    const rect = targetRef.getBoundingClientRect();
    let myTop = 0, myLeft = 0;

    const myWidth = 20;
    const myHeight = 16;
    const heightOffset = 6;
    const top = rect.y - myTop - myHeight - heightOffset;
    const left = Math.floor(rect.x - myLeft + rect.width / 2) - myWidth;
    // v hacky tooltip positioning
    useEffect(() => {
        const handler = () => setNonce(nonce + 1);
        document.addEventListener("scroll", handler);
        return () => document.removeEventListener("scroll", handler);
    });
    if (!targetRef) {
        return (<></>);
    }
    return <div onScroll={() => setNonce(nonce + 1)} style={{ backgroundColor: "white", position: "fixed", top, left, pointerEvents: 'none' }}>{value.toFixed(4)} {units}</div>;
}

export const split_color_wheel = (n: number, max = 1) => {
    let result = [];
    for (let i = 0; i < n; i++) {
        result.push((i) * (max / n));
    }
    return result;
};