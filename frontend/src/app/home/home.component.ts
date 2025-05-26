import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
@Component({
  selector: 'app-home',
  imports: [CommonModule, FormsModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {
  formData: any = {
    firearea: null,
    fwi: null,
    ffmc: null,
    dmc: null,
    dc: null,
    prec: null,
    tmax: null,
    ws: null,
    rh: null,
    fwi_prev1: null,
    fwi_prev2: null,
    rh_prev1: null,
    rh_prev2: null,
    d_vpd: null,
    pctgrowth: null,
    prevgrow: null,
  };

  prediction: string = '';
  error: string = '';

  private apiUrl = 'http://localhost:5000/'; // Flask backend URL

  constructor(private http: HttpClient) {}

  onSubmit() {
    this.http.post<any>(this.apiUrl, this.formData).subscribe({
      next: (res) => {
        console.log('Log Prediction:', res);
        this.prediction = res.prediction;
        this.error = res.error;
      },
      error: (err) => {
        console.error('Prediction failed:', err);
      },
    });
  }

  resetForm() {
    this.formData = {
      firearea: null,
      fwi: null,
      ffmc: null,
      dmc: null,
      dc: null,
      prec: null,
      tmax: null,
      ws: null,
      rh: null,
      fwi_prev1: null,
      fwi_prev2: null,
      rh_prev1: null,
      rh_prev2: null,
      d_vpd: null,
      pctgrowth: null,
      prevgrow: null,
    };
    this.prediction = '';
    this.error = '';
  }
}
