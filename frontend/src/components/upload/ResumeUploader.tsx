import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAnalysisStore } from '../../store/analysisStore';

export default function ResumeUploader() {
  const { resumeFile, setResumeFile } = useAnalysisStore();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        setResumeFile(acceptedFiles[0]);
      }
    },
    [setResumeFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
        ${isDragActive
          ? 'border-blue-400 bg-blue-500/10'
          : resumeFile
            ? 'border-green-500/50 bg-green-500/5'
            : 'border-white/10 hover:border-white/20 bg-white/[0.02]'
        }
      `}
    >
      <input {...getInputProps()} />
      {resumeFile ? (
        <div>
          <div className="text-green-400 text-lg mb-1">{resumeFile.name}</div>
          <div className="text-gray-500 text-sm">
            {(resumeFile.size / 1024).toFixed(0)} KB - Click or drop to replace
          </div>
        </div>
      ) : (
        <div>
          <div className="text-4xl mb-3 opacity-40">
            {isDragActive ? '+' : ''}
          </div>
          <div className="text-gray-300 mb-1">
            {isDragActive ? 'Drop your resume here' : 'Drag & drop your resume PDF'}
          </div>
          <div className="text-gray-500 text-sm">or click to browse (max 5MB)</div>
        </div>
      )}
    </div>
  );
}
