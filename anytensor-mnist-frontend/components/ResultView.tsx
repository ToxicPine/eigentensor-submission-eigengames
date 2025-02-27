'use client'

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { AlertCircle, CheckCircle2 } from "lucide-react"

interface ResultViewProps {
  result: { status: string; prediction?: number }
  onTryAgain: () => void
}

type AlertConfigType = {
  variant: "destructive" | "default" | null | undefined;
  className: string;
  icon: React.ReactNode;
  title: string;
  titleClassName: string;
  description: string;
  descriptionClassName: string;
}

export default function ResultView({ result, onTryAgain }: ResultViewProps) {
  const alertConfig: AlertConfigType = result.status === 'error' 
    ? {
        variant: "destructive",
        className: "max-w-md bg-red-50 border-red-200 flex flex-row items-center",
        icon: <AlertCircle className="h-8 w-8 text-red-500" />,
        title: "Error",
        titleClassName: "text-red-800 text-lg font-medium",
        description: "There was a problem processing your drawing. Please try again.",
        descriptionClassName: "text-gray-600 text-md"
      }
    : {
        variant: "default", 
        className: "max-w-md bg-green-50 border-green-200 flex flex-row items-center",
        icon: <CheckCircle2 className="h-8 w-8 text-green-600" />,
        title: "Success",
        titleClassName: "text-green-800 text-lg font-medium",
        description: `The AI guesses that this number is ${result.prediction}`,
        descriptionClassName: "text-gray-600 text-md"
      }

  return (
    <div className="flex flex-col items-center gap-4">
      <Alert variant={alertConfig.variant} className={alertConfig.className}>
        {alertConfig.icon}
        <div className="flex flex-col">
            <AlertTitle className={alertConfig.titleClassName}>
            {alertConfig.title}
            </AlertTitle>
            <AlertDescription className={alertConfig.descriptionClassName}>
            {alertConfig.description}
            </AlertDescription>
        </div>
      </Alert>
      <Button onClick={onTryAgain} className="w-full grow">Try Again</Button>
    </div>
  )
}