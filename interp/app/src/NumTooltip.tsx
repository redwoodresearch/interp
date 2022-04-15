import { useState, useEffect } from "react";

export function NumTooltip(props: { value: number, units: string, targetRef: HTMLElement; }) {
    const { targetRef, units, value } = props;
    const [nonce, setNonce] = useState(0);
    // v hacky tooltip positioning
    useEffect(() => {
        const handler = () => setNonce(nonce + 1);
        document.addEventListener("scroll", handler);
        return () => document.removeEventListener("scroll", handler);
    });
    if (!targetRef) {
        return (<></>);
    }
    const rect = targetRef.getBoundingClientRect();
    let myTop = 0, myLeft = 0;

    const myWidth = 20;
    const myHeight = 16;
    const heightOffset = 6;
    const top = rect.y - myTop - myHeight - heightOffset;
    const left = Math.floor(rect.x - myLeft + rect.width / 2) - myWidth;
    return <div onScroll={() => setNonce(nonce + 1)} style={{ backgroundColor: "white", position: "fixed", top, left, pointerEvents: 'none' }}>{value.toFixed(4)} {units}</div>;
}