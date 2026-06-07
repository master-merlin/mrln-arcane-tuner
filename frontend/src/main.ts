console.log('Bootstrapping application...');
import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app';

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => {
    console.error(err);
    const errorDiv = document.getElementById('app-error');
    if (errorDiv) {
      errorDiv.style.display = 'block';
      errorDiv.textContent = 'Bootstrap Error:\n' + (err?.stack || err);
    }
  });
