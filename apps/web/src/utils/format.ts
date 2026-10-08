export const money=(value:unknown)=>value==null||value===""?"--":Number(value).toLocaleString("en-IN",{maximumFractionDigits:2});
export const percent=(value:unknown)=>value==null?"--":`${(Number(value)*100).toFixed(1)}%`;
export const readable=(value:unknown)=>value==null||value===""?"Unavailable":String(value).replaceAll("_"," ");
export const time=(value:unknown)=>value?new Date(String(value)).toLocaleTimeString("en-IN",{timeZone:"Asia/Kolkata",hour:"2-digit",minute:"2-digit"}):"--";
