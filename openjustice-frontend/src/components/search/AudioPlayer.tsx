import React, { useRef, useState } from "react";
import { Volume2, Loader2, AlertCircle, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { realApi } from "@/lib/api/realClient";

interface AudioPlayerProps {
  text: string;
}

export default function AudioPlayer({ text }: AudioPlayerProps) {
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generateAudio = async () => {
    setLoading(true);
    setError(null);
    try {
      const blob = await realApi.generateAudio(text);
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch {
      setError("Audio generation failed");
    } finally {
      setLoading(false);
    }
  };

  const clearAudio = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-surface-50 dark:bg-surface-800 rounded-lg">
      {!audioUrl ? (
        <Button onClick={generateAudio} disabled={loading} size="sm" variant="outline">
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Generating...
            </>
          ) : (
            <>
              <Volume2 className="h-4 w-4" /> Listen
            </>
          )}
        </Button>
      ) : (
        <>
          <audio src={audioUrl} controls className="h-10" onError={() => setError("Playback failed")} />
          <Button onClick={clearAudio} size="sm" variant="ghost" aria-label="Clear audio">
            <X className="h-4 w-4" />
          </Button>
        </>
      )}
      {error && (
        <span className="flex items-center gap-1 text-sm text-red-600 dark:text-red-400">
          <AlertCircle className="h-4 w-4" /> {error}
        </span>
      )}
    </div>
  );
}
