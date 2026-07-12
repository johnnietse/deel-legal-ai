import { Scale } from "lucide-react";
import { Link } from "react-router-dom";

export function Footer() {
  return (
    <footer className="border-t border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-900">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-500">
                <Scale className="h-4 w-4 text-white" />
              </div>
              <span className="text-lg font-bold text-primary-500">OpenJustice.ai</span>
            </div>
            <p className="text-sm text-surface-500 dark:text-surface-400 max-w-md leading-relaxed">
              AI-powered legal research and worker classification platform built for the 
              Conflict Analytics Lab at Queen&apos;s University, in partnership with Deel Inc.
            </p>
          </div>

          {/* Product */}
          <div>
            <h3 className="text-sm font-semibold text-surface-900 dark:text-surface-100 mb-3">Product</h3>
            <ul className="space-y-2">
              <li><Link to="/#features" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">Features</Link></li>
              <li><Link to="/#pricing" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">Pricing</Link></li>
              <li><Link to="/chat" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">Legal Chat</Link></li>
              <li><Link to="/classify" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">Classifier</Link></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="text-sm font-semibold text-surface-900 dark:text-surface-100 mb-3">Institutional</h3>
            <ul className="space-y-2">
              <li>
                <a href="https://www.queensu.ca" target="_blank" rel="noopener noreferrer" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">
                  Queen&apos;s University
                </a>
              </li>
              <li>
                <a href="https://conflictanalyticslab.ca" target="_blank" rel="noopener noreferrer" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">
                  Conflict Analytics Lab
                </a>
              </li>
              <li>
                <a href="https://www.deel.com" target="_blank" rel="noopener noreferrer" className="text-sm text-surface-500 hover:text-primary-500 transition-colors">
                  Deel Inc.
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-surface-200 dark:border-surface-700">
          <p className="text-xs text-surface-400 text-center">
            &copy; {new Date().getFullYear()} Conflict Analytics Lab, Queen&apos;s University. All rights reserved.
            This platform is for research and educational purposes. It does not constitute legal advice.
          </p>
        </div>
      </div>
    </footer>
  );
}
