import { isOneArrayPrefixOfOther, maskToInverseIndex, range, stateToFakeTree } from "./common";
import { AttributionPath, AttributionRoot, AttribLocation, LogitRoot, PickArgs, Attributions, LocationUsedAndBase, AttribSetSpec, AttribSetState } from "./proto";
import { colorNegPos, tokenToInlineString } from "./ui_common";
import {attRootToString} from "./AttributionUISvg";

export function AttributionUISetSvg(props: { state: AttribSetState, ohInner: (...x: any[]) => void, layerNamesWithIO: string[], tokens: string[], hoveredPath: AttributionPath | null, spec: AttribSetSpec; availableWidth: number; availableHeight: number; }) {
    const { state, ohInner, tokens, layerNamesWithIO, hoveredPath, spec, availableWidth, availableHeight } = props;
    console.log("svg props", props);
    const c = {
        topAxisMargin: 4,

        maxWordWidth: 56,
        fontSize: 12,
        smallFontSize: 8,
        midFontSize: 10,

        seperation: 65,
        headMlpVerticalSeperation: 30,
        headWidth: 34,
        wordMargin: 3,
        headSpacing: 1,
        seqSpacing: 20,
        headHeight: 40,
        neuronWidth: 24,
        neuronHeight: 20,
        qkvStride: 11,
        qkvPermutation: [2, 0, 1], // show them in this order because k goes to early tokens, q goes to later tokens

        lineWidthMax: 3,
        lineWidthOnHover: 5,
        lineWidthOnUnHover: 0.2,
        backgroundOpacityFactor: 0.1,
        splineSharpness: 0.7,
        hueNeg: 0.6,
        huePos: 0.0,
        lineHoverMouseWidth: 5,
        useSplines: true,
    };

    // always show input layer
    const usedLayersMask = layerNamesWithIO.map((x, i) => (i === 0 || i === layerNamesWithIO.length - 1 ? [true, false] : [false, false]) as [boolean, boolean]);
    const usedTokensMask = tokens.map(x => false);
    state.set?.locations.forEach((loc, i) => {
        usedTokensMask[loc.token] = true;
        usedLayersMask[loc.layerWithIO][loc.isMlp ? 1 : 0] = true;
    });
    const usedLayers = range(usedLayersMask.length).filter(x => usedLayersMask[x].some(x => x));

    const usedTokens = range(usedTokensMask.length).filter(x => usedTokensMask[x]);
    console.log("usedTokens", usedTokens);
    const usedHeads = usedLayers.map((layer_idx, used_layer_idx) => usedTokens.map((tok_idx, used_tok_idx) =>
        ([] as number[])
    ));

    const usedNeurons = usedLayers.map((layer_idx, used_layer_idx) => usedTokens.map((tok_idx, used_tok_idx) =>
        ([] as number[])
    ));

    const layerIdxsToUsed = maskToInverseIndex(usedLayersMask.map(x => x.some(y => y)));
    const tokenIdxsToUsed = maskToInverseIndex(usedTokensMask);

    const seqMaxWidths = usedTokens.map(() => 0);

    const fillUsedHeadsAndNeurons = (idx: AttribLocation) => {

        const hereUsed = { layerWithIO: layerIdxsToUsed[idx.layerWithIO], token: tokenIdxsToUsed[idx.token], isMlp: idx.isMlp, headOrNeuron: -1, headReplica: -1 };
        // console.log("hereUsed", hereUsed);
        const hereHeads = usedHeads[hereUsed.layerWithIO][hereUsed.token];
        const hereNeurons = usedNeurons[hereUsed.layerWithIO][hereUsed.token];
        const hereHeadOrNeurons = hereUsed.isMlp ? hereNeurons : hereHeads;
        let matchingHeadIdx = range(hereHeadOrNeurons.length).filter(i => hereHeadOrNeurons[i] === idx.headOrNeuron)[0];
        if (matchingHeadIdx === undefined) {
            hereHeadOrNeurons.push(idx.headOrNeuron);
            matchingHeadIdx = hereHeadOrNeurons.length - 1;
        }
        hereUsed.headOrNeuron = matchingHeadIdx;

        seqMaxWidths[hereUsed.token] = Math.max(seqMaxWidths[hereUsed.token], hereHeads.length * c.headWidth, hereNeurons.length * c.neuronWidth,c.maxWordWidth);
        return hereUsed;
    };

    state.set?.locations.forEach(fillUsedHeadsAndNeurons);

    const seqPoses = [] as number[];
    let curxpos = 0;

    for (let i = 0; i < seqMaxWidths.length; i++) {
        seqPoses.push(curxpos + (seqMaxWidths[i] + c.seqSpacing) / 2);
        curxpos += seqMaxWidths[i]+ c.seqSpacing;
    }
    seqPoses.push(curxpos);

    let curypos = 0;
    const layerPoses = [] as [number, number][];
    for (let i = 0; i < usedLayersMask.length; i++) {
        const hereHeadMlp = usedLayersMask[i];
        if (hereHeadMlp[0] || hereHeadMlp[1]) {
            const hereHeadMlpPos = [-1, -1] as [number, number];
            layerPoses.push(hereHeadMlpPos);
            if (hereHeadMlp[0]) {
                hereHeadMlpPos[0] = curypos;
                curypos += c.headHeight;
            }
            if (hereHeadMlp[1]) {
                if (hereHeadMlp[0]) {
                    curypos += c.headMlpVerticalSeperation;
                }
                hereHeadMlpPos[1] = curypos;
                curypos += c.neuronHeight;
            }
            curypos += c.seperation;
        }
    }
    layerPoses.push([curypos, curypos]);

    // currently just putting words over where they should go
    // and not worrying about their widths
    const yTop = (c.fontSize + c.wordMargin) * 2 + c.headHeight + c.topAxisMargin;
    const neededHeight = yTop + Math.max(...layerPoses[layerPoses.length - 1]);
    const neededWidth = (seqPoses[seqPoses.length - 1]) + c.maxWordWidth * 2;
    const isSomethingHovered = !!hoveredPath;
    console.log("SVG pre render state", { state, usedHeads, seqPoses, seqMaxWidths, layerPoses, usedTokens, usedNeurons, usedLayersMask, usedLayers });
    // const inputSums = state.tree.length > 0 && covTreeSumPick(state.tree, [0, 0, null, 0]);

    const isLocationHovered = (hoveredPath: AttributionPath | null, location: AttribLocation) => {
        if (hoveredPath === null) return false;
        for (let pathLoc of hoveredPath) {
            if (JSON.stringify(pathLoc) === JSON.stringify(location)) {
                return true;
            }
        }
        return false;
    }; // TODO: make this work for partial locations

    const usedLocToYPos = (usedLocation: AttribLocation) => {
        return layerPoses[usedLocation.layerWithIO][usedLocation.isMlp ? 1 : 0];
    };


    const usedLocToXPos = (usedLocation: AttribLocation) => {
        // console.log("usedLocToXPos", usedLocation)
        let x = seqPoses[usedLocation.token];
        if (usedLayers[usedLocation.layerWithIO] !== 0 && usedLayers[usedLocation.layerWithIO] < layerNamesWithIO.length - 1) {
            x += 11; // HACK I don't know why I need this
            if (usedLocation.isMlp) {
                const layerSeqNeurons = usedNeurons[usedLocation.layerWithIO][usedLocation.token];
                x -= (layerSeqNeurons.length) * c.neuronWidth / 2;
                x += layerSeqNeurons.slice(0, usedLocation.headOrNeuron).length * c.neuronWidth;
            } else {
                const layerSeqHeads = usedHeads[usedLocation.layerWithIO][usedLocation.token];
                x -= ((layerSeqHeads.length-0.5) * c.headWidth) / 2;
                const prevConjoinedHeadsWidth = layerSeqHeads.slice(0, usedLocation.headOrNeuron).length * c.headWidth;
                x += prevConjoinedHeadsWidth;
                if (usedLocation.qkv !== null && usedLocation.qkv !== undefined) {
                    x += (c.qkvPermutation[usedLocation.qkv] - 1) * c.qkvStride;
                }
            }
        }
        return x;
    };

    const heads = [] as JSX.Element[];
    const neurons = [] as JSX.Element[];

    state.set?.locations.forEach((loc, nodeIdx) => {
        const nodeValue = state.set?.nodeValues[nodeIdx] as number;
        const headOrNeuron = loc.headOrNeuron;
        const usedLayer = layerIdxsToUsed[loc.layerWithIO];
        const usedToken = tokenIdxsToUsed[loc.token];
        const usedHeadOrNeuron = (loc.isMlp ? usedNeurons : usedHeads)[usedLayer][usedToken].indexOf(loc.headOrNeuron);
        const usedLoc = { layerWithIO: usedLayer, token: usedToken, headOrNeuron: usedHeadOrNeuron, isMlp: loc.isMlp };
        const x = usedLocToXPos(usedLoc);
        const y = usedLocToYPos(usedLoc);
        const areQkvHovered = [false, false, false];
        const isOutputHovered = false;
        if (usedLoc.isMlp) {
            neurons.push(<g key={`${usedLayer}-${usedToken}-${usedHeadOrNeuron}`} transform={`translate(${x},${y})`}>
                <rect x={-c.neuronWidth * 0.5} y={-c.neuronHeight * 0.5} height={c.neuronHeight} width={c.neuronWidth} fill={colorNegPos(nodeValue, c.huePos, c.hueNeg)} stroke="black" rx={1}></rect>
                <rect x={-c.neuronWidth * 0.5} y={-c.neuronHeight * 0.5} height={c.neuronHeight} width={c.neuronWidth} fill="#00000000" stroke="#00000000" ></rect>
                <text fontWeight={isOutputHovered ? "bold" : ""} transform={`translate(${0},${3})`} textAnchor="middle" fontSize="8px" {...ohInner(loc)}>{headOrNeuron}</text>
            </g>);
        } else if (loc.layerWithIO !== 0 && loc.layerWithIO !== layerNamesWithIO.length - 1) {
            heads.push(<g key={`${usedLayer}-${usedToken}-${usedHeadOrNeuron}`} transform={`translate(${x},${y})`}>
                <rect x={-c.headWidth * 0.5} y={-c.headHeight * 0.5} height={c.headHeight} width={c.headWidth} fill={colorNegPos(nodeValue, c.huePos, c.hueNeg)} stroke="black" rx={2}></rect>
                <rect x={-c.headWidth * 0.5} y={-c.headHeight * 0.5} height={c.headHeight} width={c.headWidth} fill="#00000000" stroke="#00000000" {...ohInner(loc)}></rect>
                <g transform={`translate(0,-7)`}>
                    <text fontWeight={areQkvHovered[0] ? "bold" : ""} transform={`translate(${c.qkvStride},0)`} textAnchor="middle" {...ohInner({ ...loc, qkv: 0 })}>q</text>
                    <text fontWeight={areQkvHovered[1] ? "bold" : ""} textAnchor="middle" transform={`translate(${-c.qkvStride},0)`} {...ohInner({ ...loc, qkv: 1 })}>k</text>
                    <text fontWeight={areQkvHovered[2] ? "bold" : ""} textAnchor="middle" {...ohInner({ ...loc, qkv: 2 })}>v</text>
                </g>
                <text fontWeight={isOutputHovered ? "bold" : ""} transform={`translate(${0},${8})`} textAnchor="middle" >{headOrNeuron}</text>
            <text fontSize={`${c.smallFontSize}px`} transform={`translate(${0},${ 17})`} textAnchor="middle">{nodeValue.toFixed(3)}</text>
            </g>);
        }
    });

    const lines = [] as JSX.Element[];
    const lineLabels = [] as JSX.Element[];

    const renderLine = (usedFrom: AttribLocation, usedTo: AttribLocation, baseFrom:AttribLocation,baseTo:AttribLocation,rawValue:number) => {
        // console.log({rawValue,usedFrom,usedTo,attribs})
        const value = rawValue * spec.lineWidthScale;
        let [x1, y1, x2, y2] = [
            usedLocToXPos(usedFrom),
            usedLocToYPos(usedFrom),
            usedLocToXPos(usedTo),
            usedLocToYPos(usedTo),
        ];
        if (baseFrom.layerWithIO !== 0) {
            if (usedFrom.isMlp) {
                y1 += c.neuronHeight / 2;
            } else {
                y1 += c.headHeight / 2;
            }
        }

        if (usedLayers[usedTo.layerWithIO] !== layerNamesWithIO.length - 1) {
            if (usedTo.isMlp) {
                y2 -= c.neuronHeight / 2;
            } else {
                y2 -= c.headHeight / 2;
            }
        }

        const thisHovered = false
        const yDist = y2 - y1;
        const pathString = `M${x1} ${y1}, ${c.useSplines ? `C${x1} ${y1 + yDist * c.splineSharpness}, ${x2} ${y2 - yDist * c.splineSharpness}, ${x2} ${y2}` : `L ${x2} ${y2}`}`;
        return [<g key={`${JSON.stringify(baseTo)}${JSON.stringify(baseFrom)}`}>
            <path style={{ pointerEvents: "none" }} d={pathString} stroke={colorNegPos(Math.sign(value), c.huePos, c.hueNeg)} strokeWidth={isSomethingHovered ? (thisHovered ? c.lineWidthOnHover : c.lineWidthOnUnHover) : c.lineWidthMax * Math.min(1, Math.abs(value))} fill="none" />
            {/* another thicker invisible line to make hovering easier */}
            <path d={pathString} {...ohInner(baseTo)} stroke={"#00000000"} strokeWidth={c.lineHoverMouseWidth} fill="none" />

        </g>,
        <g key={`${JSON.stringify(baseTo)}${JSON.stringify(baseFrom)}`} style={{ pointerEvents: "none" }}>
            <rect transform={`translate(${x1 - c.smallFontSize * 1.5},${y1 + 1})`} width={c.smallFontSize * 3} height={c.smallFontSize + c.wordMargin} fill="white" />
            <text fontSize={`${c.smallFontSize}px`} transform={`translate(${x1},${y1 + c.smallFontSize + 1})`} textAnchor="middle">{rawValue.toFixed(3)}</text>

            <rect transform={`translate(${x2 - c.smallFontSize * 1.5},${y2 - c.smallFontSize - c.wordMargin - 1})`} width={c.smallFontSize * 3} height={c.smallFontSize + c.wordMargin} fill="white" />
            <text fontSize={`${c.smallFontSize}px`} transform={`translate(${x2},${y2 - c.wordMargin - 1})`} textAnchor="middle">{rawValue.toFixed(3)}</text>
        </g>];
    };
    
    const baseToUsed = (base:AttribLocation)=>{
        console.log(base)
        const result = {layerWithIO:layerIdxsToUsed[base.layerWithIO], token:tokenIdxsToUsed[base.token],isMlp:base.isMlp,qkv:base.qkv} as AttribLocation
        // console.log(result)
        result.headOrNeuron = (base.isMlp?usedNeurons:usedHeads)[result.layerWithIO][result.token].indexOf(base.headOrNeuron)
        return result
    }
    
    state.set?.edges.forEach((e,i)=>{
        const {to,from,value} = e
        const toUsed = baseToUsed(to)
        const fromUsed = baseToUsed(from)
        const [line,label] = renderLine(fromUsed,toUsed,from,to,value)
        lines.push(line)
        lineLabels.push(label)
    })

    const tokenSums = tokens.map(x => 0);
    state.set?.locations.forEach((loc,i)=>{
        if (loc.layerWithIO===0){
            tokenSums[loc.token]+=state.set!.nodeValues[i]
        }
    })

    console.log("to render bits", { lines, lineLabels, heads, neurons });
    const wordRectWidth = c.maxWordWidth+c.seqSpacing
    return (
        <svg viewBox={`0 0 ${neededWidth} ${neededHeight}`} width={`${availableWidth}px`} height={`${availableHeight}px`} >
            <g transform={`translate(${c.maxWordWidth*2},${yTop})`}>

                <g className="layer-names" transform={`translate(0,${6})`}>
                    {usedLayers.map((layer_idx, used_layer_idx) => layerPoses[used_layer_idx][0] !== -1 && (<text key={layer_idx} textAnchor="end" transform={`translate(0,${layerPoses[used_layer_idx][0]})`}>{`${layer_idx !== 0 && layer_idx !== layerNamesWithIO.length - 1 ? "attn " : ""}${layerNamesWithIO[layer_idx]}`}</text>))}
                    {usedLayers.map((layer_idx, used_layer_idx) => used_layer_idx !== 0 && layerPoses[used_layer_idx][1] !== -1 && (<text key={layer_idx} textAnchor="end" transform={`translate(0,${layerPoses[used_layer_idx][1]})`}>{`mlp ${layerNamesWithIO[layer_idx]}`}</text>))}
                </g>

                <g className="tokens-and-positions" transform={`translate(0,${-c.fontSize - c.wordMargin})`} >
                    {usedTokens.map((seqIdx, used_seq_idx) => (<g key={seqIdx} >
                    <rect transform={`translate(${ seqPoses[used_seq_idx]- wordRectWidth/2},${-c.fontSize-2})`} width={wordRectWidth} height={c.fontSize + 15} fill={colorNegPos(tokenSums[seqIdx],c.huePos,c.hueNeg)} rx={2} stroke="black" />
                    <text fontWeight={isLocationHovered(hoveredPath, { layerWithIO: 0, token: seqIdx, headOrNeuron: 0, isMlp: false }) ? "bold" : ""} textAnchor="middle" transform={`translate(${seqPoses[used_seq_idx]},0)`}>{tokenToInlineString(tokens[seqIdx])}</text>
                    <text fontWeight={isLocationHovered(hoveredPath, { layerWithIO: 0, token: seqIdx, headOrNeuron: 0, isMlp: false }) ? "bold" : ""} textAnchor="middle" fontSize={c.midFontSize} transform={`translate(${seqPoses[used_seq_idx]},${9})`}>{tokenSums[seqIdx].toFixed(3)}</text>
                    
                    <text fontWeight={isLocationHovered(hoveredPath, { layerWithIO: 0, token: seqIdx, headOrNeuron: 0, isMlp: false }) ? "bold" : ""} transform={`translate(${seqPoses[used_seq_idx]},${c.fontSize + c.wordMargin+13})`} textAnchor="middle">{`${seqIdx}`}</text>
                    </g>))}
                </g>

                <g className="around-lines">{lines}</g>
                <g className="around-heads">{heads}</g>
                <g className="around-neurons">{neurons}</g>
                <g className="around-line-labels">{lineLabels}</g>
                <g transform={`translate(0,${Math.max(...layerPoses[layerPoses.length - 2]) + 6})`}>
                    {usedTokens.map((seqIdx, used_seq_idx) => spec.root && seqIdx === spec.root.data.seqIdx && (
                        <text key={seqIdx} textAnchor="middle" transform={`translate(${seqPoses[used_seq_idx]},0)`} {...ohInner([])}>{attRootToString(spec.root)}</text>))}
                </g>
            </g>
        </svg>);
}