import { useAnalysisStore } from '../../store/analysisStore';

export default function JobDescriptionInput() {
  const { jobDescription, setJobDescription } = useAnalysisStore();

  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        Job Description
      </label>
      <textarea
        value={jobDescription}
        onChange={(e) => setJobDescription(e.target.value)}
        placeholder="Paste the job description here..."
        rows={8}
        maxLength={10000}
        className="w-full bg-slate-900/80 border border-white/10 rounded-xl px-4 py-3 text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition-colors"
      />
      <div className="text-right text-xs text-gray-600 mt-1">
        {jobDescription.length} / 10,000
      </div>
    </div>
  );
}
