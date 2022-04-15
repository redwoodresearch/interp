import { useEffect, useRef, useState } from 'react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { VeryNamedTensor, PickArgs, AttribLocation, AttributionPath, AttributionBackend, LogitRoot, QueriedLogits, AttribSetSpec, AttribSetState, IMPLEMENTED_TOKEN_ROOTS } from "./proto";
import { deepPromiseAll } from "./common";
import { colorNegPos, Select, tokenToInlineString } from "./ui_common";
import ndarray from 'ndarray';
import { vocab } from "./tokens";
import { MultiHueText } from './MultiHueText';
import { AttributionUISetSvg } from './AttributionUISetSvg';

// I need this debounce / netMouseEnters bullshit because mouseenter and mouseleave callbacks aren't always called in the order I want
let debounce = null as any;

export function AttributionUISet(props: { urlState: AttribSetSpec, setUrlState: (x: AttribSetSpec) => void, options: any, attributionBackend: AttributionBackend; }) {

    const { options, urlState: spec, setUrlState: setSpec } = props;
    const Backend: AttributionBackend = props.attributionBackend as any;
    const { headNames, layerNames, tokens, neuronNames } = Backend;
    const aroundFigureDivRef = useRef(null as null | HTMLDivElement);

    const layerNamesWithIO = ["embeds", ...layerNames, "output"];

    const [hoveredPath, _setHoveredPath] = useState(null as null | AttributionPath);
    const [numOutstandingRequests, setNumOutstandingRequests] = useState(0);

    const defaultAttribRoot = { kind: "logprob", threshold: 0.1, data: { seqIdx: tokens.length - 2, tokString: tokens[tokens.length - 1], comparisonTokString: null } };
    const defaultSpec = { root: defaultAttribRoot, lineWidthScale: 1, useIGAttn: false, useIGOutput: true, showNegative: true, useActivationVsMean: false, threshold: 0.1, specificLogits: [], nonce: 0 } as AttribSetSpec;

    const [state, setState] = useState({ set: null, pathDirectionLogits: null as null | QueriedLogits } as AttribSetState);
    const [viewportAspectRatio, setViewportAspectRatio] = useState(0.5);
    const [thresholdInProgress, setThresholdInProgress] = useState(spec?.threshold ?? 0.1);

    const setStateFromSpec = (newSpec: AttribSetSpec) => {
        const promises = { start: Backend._startTree(newSpec.root, newSpec.useIGOutput, newSpec.useActivationVsMean, false), set: Backend._searchAttributionsFromStart(newSpec.threshold, newSpec.threshold, newSpec.showNegative, newSpec.useIGAttn) };
        newSpec.nonce += 1;
        deepPromiseAll(promises).then((results) => {
            const { start, set } = results;
            setSpec(newSpec);
            setState({ set: set, pathDirectionLogits: null, nonce: newSpec.nonce });
            setThresholdInProgress(newSpec.threshold);
        });
    };

    useEffect(() => {
        if (spec) {
            setStateFromSpec(spec);
        }
    }, [Backend]);

    if (!spec) {
        setStateFromSpec(defaultSpec);
        return (<p>Setting up attribution ui</p>);
    }
    if (state.nonce !== spec.nonce) {
        return (<p>Spec and state out of sync</p>);
    }


    const removeFocus = () => {
    };

    const onMouseLeave = () => {
        if (debounce !== null) {
            clearTimeout(debounce);
        }
        debounce = setTimeout(() => removeFocus(), 100);
    };

    const onMouseEnter = (loc: AttribLocation) => {
        if (debounce !== null) {
            clearTimeout(debounce);
            debounce = null;
        }
        console.log("SETTING HOVERED LOC", loc);
    };

    // console.log({ hoverReleventSeqIdxs, tokens, hoveredPath });
    const inputSums = tokens.map(x => 0);
    state.set?.locations.forEach((loc, i) => {
        if (loc.layerWithIO === 0) {
            inputSums[loc.token] += state.set!.nodeValues[i];
        }
    });
    const usedTokensStridedArr = [...tokens.map((_, i) => spec.root.data.seqIdx === i), ...inputSums]; //@TODO show used tokens even though that's only computed in inner component?

    const usedTokensVNT = { dim_names: ["UNK", "used_tokens"], dim_types: ["UNK", "seq"], dim_idx_names: [["loss_on", "attribution_from"], tokens], tensor: ndarray(usedTokensStridedArr, [3, tokens.length]), units: "shown below", title: "shown tokens" };

    const ohInner = (loc: AttribLocation) => {
        return {
            onMouseLeave: onMouseLeave,
            onMouseEnter: () => {
                onMouseEnter(loc);
            },
            onClick: () => {
                console.log("CLICK", loc);
            }
        };
    };

    const availableWidth = aroundFigureDivRef.current ? aroundFigureDivRef.current.getBoundingClientRect().width : 600;

    return (<div className="box border_green" style={{ marginTop: "12px" }}>
        <div className="section_title"> Path Attribution </div>
        <div>
            {numOutstandingRequests > 0 ? <span style={{ backgroundColor: "yellow" }}>Waiting for Server</span> : <span>​</span>}
        </div>
        <MultiHueText vnt={usedTokensVNT as VeryNamedTensor} hover={[null, null]} setFocus={(dims: PickArgs) => {// TODO make this work for hovering anything
            if (dims[1] !== null) {
                setStateFromSpec({ ...spec, root: { kind: "logprob", data: { seqIdx: dims[1], tokString: Backend.tokens[Math.min(Backend.tokens.length - 1, dims[1] + 1)], comparisonTokString: null }, threshold: spec.threshold } });
            }
        }} setHover={(dims: PickArgs) => {
            console.log("HOVERING");
        }} highlight={[null, null]}
        />
        <div className="attribution_chart" >
            <div ref={aroundFigureDivRef} tabIndex={0} className="attributionuiaroundsvg" onKeyDown={(e) => {
                console.log("KEYDOWN", e);
                if (e.altKey) {
                    const MAC_ALT_NUMBER_KEYS = `o¡™£¢∞§¶•ª`;
                    const keyInt = parseInt(e.key) || MAC_ALT_NUMBER_KEYS.indexOf(e.key);
                    if (keyInt && keyInt !== -1) {
                        setStateFromSpec({ ...spec, threshold: parseFloat(Math.exp(1.1 - keyInt).toPrecision(3)) }); // 3 to 0.0004, geometrically by intervals of e
                    }
                }
            }} onClick={(e) => ((e.target as HTMLElement).closest(".attributionuiaroundsvg") as any).focus()} style={{ outline: "none" }}>
                <TransformWrapper limitToBounds={false} minScale={0.2} wheel={{ step: 0.1, activationKeys: ["Shift", "Control", "ctrl", "shft", "Alt", "alt"] }} doubleClick={{ disabled: true }} >
                    <TransformComponent>
                        <AttributionUISetSvg state={state} ohInner={ohInner} layerNamesWithIO={layerNamesWithIO} tokens={tokens} hoveredPath={hoveredPath} spec={spec} availableWidth={availableWidth} availableHeight={availableWidth * viewportAspectRatio} />
                    </TransformComponent>
                </TransformWrapper>
            </div>
        </div>
        <div style={{ display: "flex", flexDirection: "row" }}>
            <div className="box border_green">
                <label style={{ margin: "0" }}>Add Token To Show Logprobs For</label>
                <Select sortedWords={vocab} value={null} onChange={(s: string) => setStateFromSpec({ ...spec, specificLogits: [...spec.specificLogits, s] })} defaultTopWords={tokens} placeholder={"Add Token"}></Select>
                {spec.specificLogits.length > 0 && <div style={{ display: "flex", flexDirection: "row" }}>{spec.specificLogits.map((x, i) => (<span key={i} style={{ padding: "1px 3px", margin: "4px 2px", border: "1px solid black", cursor: "pointer", borderRadius: "3px" }} onClick={() => {
                    const newSpecificLogits = [...spec.specificLogits];
                    newSpecificLogits.splice(i, 1);
                    setStateFromSpec({ ...spec, specificLogits: newSpecificLogits });
                }}>✖{tokenToInlineString(x)}</span>))}</div>}
                {spec.root && (IMPLEMENTED_TOKEN_ROOTS[spec.root.kind]) && <>
                    <label style={{ margin: "0" }}>Target Token</label>
                    <Select sortedWords={vocab} value={(spec.root.data as LogitRoot).tokString} onChange={(s: string) => setStateFromSpec({ ...spec, root: { kind: spec.root.kind, data: { ...spec.root.data, tokString: s }, threshold: spec.threshold } })} defaultTopWords={tokens} placeholder={"Target Token"}></Select>
                    <label style={{ margin: "0" }}>Comparison Token</label>
                    <Select sortedWords={vocab} value={(spec.root.data as LogitRoot).comparisonTokString} onChange={(s: string) => setStateFromSpec({ ...spec, root: { kind: spec.root.kind, data: { ...spec.root.data, comparisonTokString: s }, threshold: spec.threshold } })} defaultTopWords={tokens} showClearButton={true} placeholder={"Comparison Token"}></Select></>}
            </div>
            <div className="box border_green">
                <div className="box_title"> Logits at this position: </div>

                <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", minHeight: "500px" }}>
                    {["top", "bottom", "specific"].map((wordListName, i) => (
                        <div style={{ margin: "0 5px", display: "flex", flexDirection: "column" }}>
                            <span>{wordListName.replace(/^\w/, (c) => c.toUpperCase())}: (logits) </span>
                            {state.pathDirectionLogits ?
                                (state.pathDirectionLogits as any)[wordListName].values.map((val: number, i: number) => {
                                    return (<span style={{ width: "100%", backgroundColor: colorNegPos(val / 10 /* HACK /10 just because */) }}>
                                        <span style={{ display: "block", float: "left" }}> {tokenToInlineString((state.pathDirectionLogits as any)[wordListName].words[i])}</span>
                                        <span style={{ display: "block", float: "right" }}>{val.toPrecision(3)}</span>
                                    </span>);
                                }) : <span>No data</span>}


                        </div>))}
                </div>
            </div>

            <div className="box border_green" title="set threshold with the keyboard shortcut Alt+Number">
                <div className="box_title"> Display options: </div>
                <table>
                    <tr>
                        <td>Line visibility threshold &nbsp;</td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" step="0.1" value={spec.threshold} onChange={(e) => setStateFromSpec({ ...spec, threshold: parseFloat(e.target.value) })} /></td>
                    </tr>
                    <tr>
                        <td>Line width scale </td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" value={spec.lineWidthScale} onChange={(e) => setStateFromSpec({ ...spec, lineWidthScale: parseFloat(e.target.value) })} /></td>
                    </tr>
                    <tr>
                        <td>Aspect ratio</td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" value={viewportAspectRatio} onChange={(e) => setViewportAspectRatio(parseFloat(e.target.value))}></input></td>
                    </tr>

                    <tr>
                        <td> Integrate gradients for attn probs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={spec.useIGAttn} onClick={() => setStateFromSpec({ ...spec, useIGAttn: !spec.useIGAttn })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Integrate gradients for loss log probs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={spec.useIGOutput} onClick={() => setStateFromSpec({ ...spec, useIGOutput: !spec.useIGOutput })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Show negative attribution &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={spec.showNegative} onClick={() => setStateFromSpec({ ...spec, showNegative: !spec.showNegative })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Subtract Dataset Mean from Outputs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={spec.useActivationVsMean} onClick={() => setStateFromSpec({ ...spec, useActivationVsMean: !spec.useActivationVsMean })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    {spec.root && (IMPLEMENTED_TOKEN_ROOTS[spec.root.kind]) &&
                        (<tr><td> <span style={{ fontWeight: spec.root.kind === "logprob" ? "bold" : "" }}>Logprobs</span> or <span style={{ fontWeight: spec.root.kind === "logit" ? "bold" : "" }}>Logits</span>? &nbsp; </td>
                            <td>
                                <label className="switch">
                                    <input type="checkbox" checked={spec.root.kind === "logit"} onClick={() => setStateFromSpec({ ...spec, root: { ...spec.root, kind: (spec.root.kind === "logprob" ? "logit" : "logprob") } })} />
                                    <span className="slider round"></span>
                                </label>
                            </td></tr>)}

                </table>

            </div>


        </div>
    </div >);
};

