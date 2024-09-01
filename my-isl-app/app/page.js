'use client'
import Image from "next/image";
import * as tf from '@tensorflow/tfjs'
import { useEffect } from "react";
import ISLTranslator from "@/components/ISLTranslator";
export default function Home() {
  
  return (
    <>
    <ISLTranslator/>
    </>
  );
}
