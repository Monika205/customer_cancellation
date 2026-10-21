                "Usage Difference",
                "Total Spend",
                "Number of Penalties",
                "Service Tickets",
                "Service Rating",
                "Customer Tenure",
                "Start Month",
                "Start Week",
                "Penalty Level",
                "Fee Band",
                "Tenure Group"
            ],
            "Value": [
                territory,
                device,
                plan,
                str(promotion),
                f"{base_fee:.2f}",
                f"{extra_spend:.2f}",
                f"{current_usage:.2f}",
                f"{prior_usage:.2f}",
                f"{usage_difference:.2f}",
                f"{total_spend:.2f}",
                penalties,
                tickets,
                rating,
                tenure,
                month,
                week,
                penalty_level,
                str(fee_band),
                tenure_group
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ---------------- DOWNLOAD REPORT ----------------
        result_text = "Churn" if prediction == 1 else "No Churn"

        report = pd.DataFrame([{
            "Prediction": result_text,
            "Churn Probability (%)": round(probability * 100, 2),
            "Territory": territory,
            "DeviceType": device,
            "PlanType": plan,
            "Promotion": promotion,
            "CustomerTenure": tenure,
            "BaseFee": base_fee,
            "AdditionalFeatureSpend": extra_spend
        }])

        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.info(
            "Check that the saved model was trained using the same "
            "23 features, feature names, categories and preprocessing."
        )


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    '<p class="small-note">Customer Churn Intelligence · '
    'Machine Learning Assessment Project · Predictions are decision '
    'support estimates, not guarantees.</p>',
    unsafe_allow_html=True
)
